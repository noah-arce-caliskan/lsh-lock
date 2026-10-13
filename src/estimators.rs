//! Several entropy estimators for subsets of a binary template, to be minimised over (reviewer
//! request; see `entropy-estimators.md` in the sample-correct workspace).
//!
//! All estimators look at the templates restricted to one subset `S` (`s = |S|` bits).
//! Population-level counts use one representative reading per class; model-based estimators are
//! fitted on the readings of a training half of the classes and evaluated on the other half.
//!
//! | name            | what it computes                                                                 |
//! |-----------------|----------------------------------------------------------------------------------|
//! | `daugman`       | dF * min(-log2 m, -log2(1-m)) from all cross-class pairs (the paper's test)        |
//! | `daugman_id_min`| the same test per class (that class against all others), minimum over classes   |
//! | `daugman_id_p05`| ... 5th percentile over classes                                                  |
//! | `betabin_h2`    | -log2 P[X = 0] for a beta-binomial fitted to the cross-class distance counts      |
//! | `mcv_sum`       | NIST SP 800-90B most-common-value bound per bit (99% upper bound), summed        |
//! | `spectral`      | participation ratio of the bit correlation matrix (debiased) * mean per-bit H_inf  |
//! | `chowliu_hinf`  | -log2 of the mode probability of a Chow-Liu tree (exact by max-product)            |
//! | `guess_indep`   | guesswork of held-out classes under an independent-bit prior (Monte Carlo rank)    |
//! | `guess_tree`    | the same under the Chow-Liu prior                                                |
//!
//! Guesswork: with ranks r_(1) <= ... <= r_(T) of the T held-out classes, the empirical
//! P[rank <= r_(i)] = i/T and a min-entropy h requires P[success in q guesses] <= q 2^-h, so
//! h_est = min_i (log2 r_(i) - log2(i/T)). Ranks are estimated as in Dell'Amico & Filippone
//! (CCS 2015): rank(w) ≈ mean over model samples x of 1{p(x) > p(w)} / p(x).
//! Model-based estimators are heavier and run on the `heavy` subsets with the lowest Daugman value.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::entropy::Template;

/// Templates grouped by class, with a train/test split of the classes.
pub struct Dataset {
    /// Templates as 0/1 bytes, sorted by (class, file order).
    pub data: Vec<Vec<u8>>,
    pub class_of: Vec<usize>,
    pub class_names: Vec<String>,
    /// First template of each class.
    pub reps: Vec<usize>,
    pub train_class: Vec<bool>,
}

impl Dataset {
    /// `train_frac` of the classes (chosen with `seed`) are used to fit models.
    pub fn new(mut templates: Vec<Template>, train_frac: f64, seed: u64) -> Self {
        templates.sort_by(|a, b| a.class.cmp(&b.class));
        let mut class_names: Vec<String> = Vec::new();
        let mut class_of = Vec::new();
        let mut reps = Vec::new();
        for (i, t) in templates.iter().enumerate() {
            if class_names.last() != Some(&t.class) {
                class_names.push(t.class.clone());
                reps.push(i);
            }
            class_of.push(class_names.len() - 1);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut order: Vec<usize> = (0..class_names.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.gen_range(0..=i));
        }
        let n_train = ((class_names.len() as f64) * train_frac).round() as usize;
        let mut train_class = vec![false; class_names.len()];
        for &c in &order[..n_train] {
            train_class[c] = true;
        }
        Dataset {
            data: templates.into_iter().map(|t| t.data).collect(),
            class_of,
            class_names,
            reps,
            train_class,
        }
    }

    /// A dataset with the same classes and readings per class but known full entropy: uniform
    /// class templates, each reading flipping every bit independently with the rate that matches
    /// this dataset's mean genuine distance. Estimators should return ~`s` bits on it; how far
    /// below `s` they fall is their sample-size floor on data of this shape.
    pub fn null_like(&self, seed: u64) -> Dataset {
        let (mut sum, mut cnt) = (0.0, 0.0);
        for i in 0..self.data.len() {
            for j in i + 1..self.data.len() {
                if self.class_of[i] == self.class_of[j] {
                    let d = self.data[i]
                        .iter()
                        .zip(&self.data[j])
                        .filter(|(a, b)| a != b)
                        .count();
                    sum += d as f64 / self.data[i].len() as f64;
                    cnt += 1.0;
                }
            }
        }
        let gen = if cnt > 0.0 { sum / cnt } else { 0.0 };
        let flip = (1.0 - (1.0 - 2.0 * gen).max(0.0).sqrt()) / 2.0; // 2f(1-f) = gen
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
        let n = self.data[0].len();
        let bases: Vec<Vec<u8>> = (0..self.class_names.len())
            .map(|_| (0..n).map(|_| rng.gen::<bool>() as u8).collect())
            .collect();
        let data = self
            .class_of
            .iter()
            .map(|&c| {
                bases[c]
                    .iter()
                    .map(|&b| b ^ (rng.gen::<f64>() < flip) as u8)
                    .collect()
            })
            .collect();
        Dataset {
            data,
            class_of: self.class_of.clone(),
            class_names: self.class_names.clone(),
            reps: self.reps.clone(),
            train_class: self.train_class.clone(),
        }
    }

    fn packed(&self, sub: &[usize]) -> Vec<Vec<u64>> {
        let words = sub.len().div_ceil(64);
        self.data
            .iter()
            .map(|t| {
                let mut v = vec![0u64; words];
                for (j, &x) in sub.iter().enumerate() {
                    v[j / 64] |= (t[x] as u64 & 1) << (j % 64);
                }
                v
            })
            .collect()
    }
}

// ---------------------------------------------------------------- math helpers

/// ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Daugman's estimate from the mean and variance of fractional distances.
pub fn daugman(mean: f64, var: f64) -> f64 {
    if var <= 0.0 || mean <= 0.0 || mean >= 1.0 {
        return f64::NAN;
    }
    let df = mean * (1.0 - mean) / var;
    df * f64::min(-mean.log2(), -(1.0 - mean).log2())
}

/// -log2 P[X = 0] for X ~ BetaBinomial(s, a, b) matched to the mean and variance of fractional
/// distances (method of moments). Falls back to the binomial when there is no overdispersion.
pub fn betabinomial_h2(s: usize, mean: f64, var: f64) -> f64 {
    let sf = s as f64;
    if mean <= 0.0 || mean >= 1.0 || s < 2 {
        return f64::NAN;
    }
    // Var[X] = s m(1-m)(1 + (s-1) rho) with X = s * fractional distance
    let rho = ((var * sf * sf) / (sf * mean * (1.0 - mean)) - 1.0) / (sf - 1.0);
    if rho <= 1e-12 {
        return -sf * (1.0 - mean).log2();
    }
    let rho = rho.min(1.0 - 1e-12);
    let (a, b) = (mean * (1.0 / rho - 1.0), (1.0 - mean) * (1.0 / rho - 1.0));
    // P[X=0] = B(a, b+s) / B(a, b) = Γ(b+s) Γ(a+b) / (Γ(b) Γ(a+b+s))
    let ln_p = ln_gamma(b + sf) + ln_gamma(a + b) - ln_gamma(b) - ln_gamma(a + b + sf);
    -ln_p / std::f64::consts::LN_2
}

/// NIST SP 800-90B most-common-value estimate for one binary column (99% upper bound).
pub fn mcv_bit(ones: usize, total: usize) -> f64 {
    let p = ones.max(total - ones) as f64 / total as f64;
    let pu = (p + 2.576 * (p * (1.0 - p) / (total as f64 - 1.0)).sqrt()).min(1.0);
    -pu.log2()
}

// ---------------------------------------------------------------- Chow-Liu tree

/// Tree-structured distribution over `s` binary variables (Laplace-smoothed).
pub struct ChowLiu {
    /// Variables in an order where every parent precedes its children; `parent[order[0]]` is None.
    pub order: Vec<usize>,
    pub parent: Vec<Option<usize>>,
    /// log2 P(x_v = 1 | x_parent = a) for a in {0, 1} (root: index 0 holds log2 P(x_root = 1)).
    lp1: Vec<[f64; 2]>,
}

impl ChowLiu {
    /// Fits on rows of 0/1 values (each of length s).
    pub fn fit(rows: &[Vec<u8>]) -> Self {
        let s = rows[0].len();
        let nf = rows.len() as f64;
        let ones: Vec<f64> = (0..s)
            .map(|j| rows.iter().filter(|r| r[j] == 1).count() as f64)
            .collect();
        let mi = |i: usize, j: usize| -> f64 {
            let mut c = [[1.0f64; 2]; 2]; // Laplace
            for r in rows {
                c[r[i] as usize][r[j] as usize] += 1.0;
            }
            let tot = nf + 4.0;
            let pi = [(c[0][0] + c[0][1]) / tot, (c[1][0] + c[1][1]) / tot];
            let pj = [(c[0][0] + c[1][0]) / tot, (c[0][1] + c[1][1]) / tot];
            let mut m = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    let p = c[a][b] / tot;
                    m += p * (p / (pi[a] * pj[b])).log2();
                }
            }
            m
        };
        // Prim's algorithm for the maximum-MI spanning tree
        let mut in_tree = vec![false; s];
        let mut best = vec![f64::NEG_INFINITY; s];
        let mut parent = vec![None; s];
        let mut order = Vec::with_capacity(s);
        let mut cur = 0;
        in_tree[0] = true;
        order.push(0);
        for _ in 1..s {
            for v in 0..s {
                if !in_tree[v] {
                    let w = mi(cur, v);
                    if w > best[v] {
                        best[v] = w;
                        parent[v] = Some(cur);
                    }
                }
            }
            let next = (0..s)
                .filter(|&v| !in_tree[v])
                .max_by(|&a, &b| best[a].total_cmp(&best[b]))
                .unwrap();
            in_tree[next] = true;
            order.push(next);
            cur = next;
        }
        let lp1 = (0..s)
            .map(|v| match parent[v] {
                None => {
                    let p = (ones[v] + 1.0) / (nf + 2.0);
                    [p.log2(), p.log2()]
                }
                Some(u) => {
                    let mut c = [[1.0f64; 2]; 2];
                    for r in rows {
                        c[r[u] as usize][r[v] as usize] += 1.0;
                    }
                    [
                        (c[0][1] / (c[0][0] + c[0][1])).log2(),
                        (c[1][1] / (c[1][0] + c[1][1])).log2(),
                    ]
                }
            })
            .collect();
        ChowLiu { order, parent, lp1 }
    }

    fn lp(&self, v: usize, x_v: u8, x_par: u8) -> f64 {
        let l1 = self.lp1[v][x_par as usize];
        if x_v == 1 {
            l1
        } else {
            (1.0 - 2f64.powf(l1)).max(1e-300).log2()
        }
    }

    /// log2 probability of a full assignment.
    pub fn log2_prob(&self, x: &[u8]) -> f64 {
        (0..x.len())
            .map(|v| match self.parent[v] {
                None => self.lp(v, x[v], 0),
                Some(u) => self.lp(v, x[v], x[u]),
            })
            .sum()
    }

    /// log2 of the largest probability of any assignment (max-product over the tree).
    pub fn log2_mode(&self) -> f64 {
        let s = self.order.len();
        // best[v][a] = max over the subtree of v of log2 P(subtree | x_v = a), children first
        let mut best = vec![[0.0f64; 2]; s];
        for &v in self.order.iter().rev() {
            if let Some(u) = self.parent[v] {
                let m = [0u8, 1].map(|pa| {
                    [0u8, 1]
                        .iter()
                        .map(|&a| self.lp(v, a, pa) + best[v][a as usize])
                        .fold(f64::NEG_INFINITY, f64::max)
                });
                best[u][0] += m[0];
                best[u][1] += m[1];
            }
        }
        let r = self.order[0];
        [0u8, 1]
            .iter()
            .map(|&a| self.lp(r, a, 0) + best[r][a as usize])
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn sample<R: Rng>(&self, rng: &mut R, out: &mut [u8]) {
        for &v in &self.order {
            let par = self.parent[v].map(|u| out[u]).unwrap_or(0);
            out[v] = (rng.gen::<f64>() < 2f64.powf(self.lp1[v][par as usize])) as u8;
        }
    }
}

/// Independent-bit model (Laplace-smoothed marginals) in the same interface.
pub struct Indep {
    lp1: Vec<f64>,
}

impl Indep {
    pub fn fit(rows: &[Vec<u8>]) -> Self {
        let s = rows[0].len();
        let nf = rows.len() as f64;
        Indep {
            lp1: (0..s)
                .map(|j| {
                    ((rows.iter().filter(|r| r[j] == 1).count() as f64 + 1.0) / (nf + 2.0)).log2()
                })
                .collect(),
        }
    }
    fn log2_prob(&self, x: &[u8]) -> f64 {
        x.iter()
            .zip(&self.lp1)
            .map(|(&b, &l)| {
                if b == 1 {
                    l
                } else {
                    (1.0 - 2f64.powf(l)).log2()
                }
            })
            .sum()
    }
    fn sample<R: Rng>(&self, rng: &mut R, out: &mut [u8]) {
        for (o, &l) in out.iter_mut().zip(&self.lp1) {
            *o = (rng.gen::<f64>() < 2f64.powf(l)) as u8;
        }
    }
}

/// log2 of the estimated rank (1-based) of each target under a model, from `m` model samples.
fn mc_log2_ranks(
    log2_prob: &dyn Fn(&[u8]) -> f64,
    sample: &dyn Fn(&mut ChaCha8Rng, &mut [u8]),
    targets: &[Vec<u8>],
    m: usize,
    seed: u64,
) -> Vec<f64> {
    let s = targets[0].len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut lps: Vec<f64> = (0..m)
        .map(|_| {
            let mut x = vec![0u8; s];
            sample(&mut rng, &mut x);
            log2_prob(&x)
        })
        .collect();
    lps.sort_by(|a, b| b.total_cmp(a)); // most probable first
                                        // suffix-free: cumulative sum of 2^-lp over samples more probable than the target, in log2
    let mut cum = Vec::with_capacity(m);
    let mut acc = f64::NEG_INFINITY; // log2 of running sum
    for &lp in &lps {
        let term = -lp;
        acc = if acc == f64::NEG_INFINITY {
            term
        } else {
            let (hi, lo) = if acc > term { (acc, term) } else { (term, acc) };
            hi + (1.0 + 2f64.powf(lo - hi)).log2()
        };
        cum.push(acc);
    }
    targets
        .iter()
        .map(|t| {
            let lt = log2_prob(t);
            let k = lps.partition_point(|&lp| lp > lt); // samples strictly more probable
            let est = if k == 0 {
                f64::NEG_INFINITY
            } else {
                cum[k - 1] - (m as f64).log2()
            };
            // rank = (#more probable) + 1
            let r = if est == f64::NEG_INFINITY {
                0.0
            } else {
                2f64.powf(est)
            };
            (r + 1.0).log2()
        })
        .collect()
}

/// min_i (log2 r_(i) - log2(i/T)) over ascending ranks.
pub fn guesswork_bound(mut log2_ranks: Vec<f64>) -> f64 {
    log2_ranks.sort_by(|a, b| a.total_cmp(b));
    let t = log2_ranks.len() as f64;
    log2_ranks
        .iter()
        .enumerate()
        .map(|(i, &lr)| lr - ((i + 1) as f64 / t).log2())
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------- per-subset estimators

pub const CHEAP: [&str; 6] = [
    "daugman",
    "daugman_id_min",
    "daugman_id_p05",
    "betabin_h2",
    "mcv_sum",
    "spectral",
];
pub const HEAVY: [&str; 3] = ["chowliu_hinf", "guess_indep", "guess_tree"];

/// The cheap estimators for one subset, in `CHEAP` order.
pub fn cheap(ds: &Dataset, sub: &[usize]) -> [f64; 6] {
    let s = sub.len();
    let sf = s as f64;
    let packed = ds.packed(sub);
    let nc = ds.class_names.len();
    let (mut sum, mut sq, mut cnt) = (0.0, 0.0, 0.0);
    let mut per = vec![(0.0f64, 0.0f64, 0.0f64); nc];
    let t = packed.len();
    for i in 0..t {
        for j in i + 1..t {
            let (ci, cj) = (ds.class_of[i], ds.class_of[j]);
            if ci == cj {
                continue;
            }
            let d: u32 = packed[i]
                .iter()
                .zip(&packed[j])
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            let f = d as f64 / sf;
            sum += f;
            sq += f * f;
            cnt += 1.0;
            for c in [ci, cj] {
                per[c].0 += f;
                per[c].1 += f * f;
                per[c].2 += 1.0;
            }
        }
    }
    let mean = sum / cnt;
    let var = sq / cnt - mean * mean;
    let mut per_id: Vec<f64> = per
        .iter()
        .filter(|p| p.2 > 1.0)
        .map(|&(a, b, c)| {
            let m = a / c;
            daugman(m, b / c - m * m)
        })
        .filter(|x| x.is_finite())
        .collect();
    per_id.sort_by(|a, b| a.total_cmp(b));
    let id_min = per_id.first().copied().unwrap_or(f64::NAN);
    let id_p05 = per_id
        .get(((per_id.len() as f64) * 0.05) as usize)
        .copied()
        .unwrap_or(f64::NAN);

    // one representative per class
    let reps: Vec<Vec<u8>> = ds
        .reps
        .iter()
        .map(|&r| sub.iter().map(|&x| ds.data[r][x]).collect())
        .collect();
    let nr = reps.len();
    let ones: Vec<usize> = (0..s)
        .map(|j| reps.iter().filter(|r| r[j] == 1).count())
        .collect();
    let mcv: f64 = ones.iter().map(|&o| mcv_bit(o, nr)).sum();

    // participation ratio of the correlation matrix, debiased by the 1/N null expectation
    let cols: Vec<Vec<f64>> = (0..s)
        .map(|j| {
            let p = ones[j] as f64 / nr as f64;
            let sd = (p * (1.0 - p)).sqrt();
            reps.iter()
                .map(|r| {
                    if sd > 0.0 {
                        (r[j] as f64 - p) / sd
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let live: Vec<usize> = (0..s).filter(|&j| ones[j] > 0 && ones[j] < nr).collect();
    let mut off2 = 0.0;
    for (a, &i) in live.iter().enumerate() {
        for &j in &live[a + 1..] {
            let c: f64 = cols[i]
                .iter()
                .zip(&cols[j])
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / nr as f64;
            off2 += (c * c - 1.0 / nr as f64).max(0.0);
        }
    }
    let l = live.len() as f64;
    let pr = l * l / (l + 2.0 * off2);
    let per_bit: f64 = (0..s)
        .map(|j| {
            let p = ones[j] as f64 / nr as f64;
            -p.max(1.0 - p).log2()
        })
        .sum::<f64>()
        / sf;
    [
        daugman(mean, var),
        id_min,
        id_p05,
        betabinomial_h2(s, mean, var),
        mcv,
        pr * per_bit,
    ]
}

/// The heavy (model-based) estimators for one subset, in `HEAVY` order.
pub fn heavy(ds: &Dataset, sub: &[usize], samples: usize, seed: u64) -> [f64; 3] {
    let restrict = |i: usize| -> Vec<u8> { sub.iter().map(|&x| ds.data[i][x]).collect() };
    let train: Vec<Vec<u8>> = (0..ds.data.len())
        .filter(|&i| ds.train_class[ds.class_of[i]])
        .map(restrict)
        .collect();
    let test: Vec<Vec<u8>> = ds
        .reps
        .iter()
        .filter(|&&r| !ds.train_class[ds.class_of[r]])
        .map(|&r| restrict(r))
        .collect();
    let tree = ChowLiu::fit(&train);
    let ind = Indep::fit(&train);
    let g_ind = guesswork_bound(mc_log2_ranks(
        &|x| ind.log2_prob(x),
        &|r, o| ind.sample(r, o),
        &test,
        samples,
        seed,
    ));
    let g_tree = guesswork_bound(mc_log2_ranks(
        &|x| tree.log2_prob(x),
        &|r, o| tree.sample(r, o),
        &test,
        samples,
        seed ^ 0x9e37,
    ));
    [-tree.log2_mode(), g_ind, g_tree]
}

pub struct SuiteResult {
    pub cheap: Vec<[f64; 6]>,
    /// (subset index, values) for the heavy subsets.
    pub heavy: Vec<(usize, [f64; 3])>,
}

/// Runs the cheap estimators on every subset and the heavy ones on the `n_heavy` subsets with
/// the lowest Daugman value.
pub fn run_suite(
    ds: &Dataset,
    subsets: &[Vec<usize>],
    n_heavy: usize,
    samples: usize,
    seed: u64,
) -> SuiteResult {
    let cheap: Vec<[f64; 6]> = subsets.par_iter().map(|s| cheap(ds, s)).collect();
    let mut idx: Vec<usize> = (0..subsets.len()).collect();
    idx.sort_by(|&a, &b| cheap[a][0].total_cmp(&cheap[b][0]));
    idx.truncate(n_heavy);
    let heavy = idx
        .par_iter()
        .map(|&i| {
            (
                i,
                heavy(ds, &subsets[i], samples, seed.wrapping_add(i as u64)),
            )
        })
        .collect();
    SuiteResult { cheap, heavy }
}

/// Unions of pairs of subsets: the `top` pairs with the largest overlap among the first `pool`
/// subsets, then `random` uniformly chosen pairs.
pub fn pair_unions(
    subsets: &[Vec<usize>],
    pool: usize,
    top: usize,
    random: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let pool = pool.min(subsets.len());
    let n = subsets.iter().flatten().max().map(|&m| m + 1).unwrap_or(0);
    let masks: Vec<Vec<u64>> = subsets[..pool]
        .iter()
        .map(|s| {
            let mut v = vec![0u64; n.div_ceil(64)];
            for &x in s {
                v[x / 64] |= 1 << (x % 64);
            }
            v
        })
        .collect();
    let mut pairs: Vec<(u32, usize, usize)> = (0..pool)
        .into_par_iter()
        .flat_map_iter(|i| {
            let masks = &masks;
            (i + 1..pool).map(move |j| {
                let o: u32 = masks[i]
                    .iter()
                    .zip(&masks[j])
                    .map(|(a, b)| (a & b).count_ones())
                    .sum();
                (o, i, j)
            })
        })
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let union = |i: usize, j: usize| -> Vec<usize> {
        let mut u: Vec<usize> = subsets[i].iter().chain(&subsets[j]).copied().collect();
        u.sort_unstable();
        u.dedup();
        u
    };
    let mut out: Vec<Vec<usize>> = pairs
        .iter()
        .take(top)
        .map(|&(_, i, j)| union(i, j))
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..random {
        let i = rng.gen_range(0..subsets.len());
        let mut j = rng.gen_range(0..subsets.len());
        while j == i {
            j = rng.gen_range(0..subsets.len());
        }
        out.push(union(i, j));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        for n in 1..20u64 {
            let f: f64 = (1..n).map(|k| (k as f64).ln()).sum();
            assert!((ln_gamma(n as f64) - f).abs() < 1e-9, "n={n}");
        }
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-9);
    }

    #[test]
    fn betabinomial_reduces_to_binomial_and_drops_with_overdispersion() {
        let s = 70;
        let m = 0.5;
        let var_bin = m * (1.0 - m) / s as f64;
        assert!((betabinomial_h2(s, m, var_bin) - 70.0).abs() < 1e-6);
        let h_over = betabinomial_h2(s, m, 3.0 * var_bin);
        assert!(h_over < 70.0 && h_over > 0.0, "{h_over}");
        // Daugman with dF = s/3 is s/3 bits; the beta-binomial tail is heavier than that binomial
        assert!((daugman(m, 3.0 * var_bin) - 70.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mcv_is_near_one_bit_for_a_fair_large_sample_and_zero_for_constant() {
        // the 99% upper bound costs ~0.012 bits at N = 100k
        let h = mcv_bit(50_000, 100_000);
        assert!(h > 0.98 && h < 1.0, "{h}");
        assert!(mcv_bit(10, 10) == 0.0);
    }

    fn rows(n: usize, s: usize, f: impl Fn(&mut ChaCha8Rng) -> Vec<u8>) -> Vec<Vec<u8>> {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let _ = s;
        (0..n).map(|_| f(&mut rng)).collect()
    }

    #[test]
    fn chow_liu_mode_entropy_on_independent_and_copied_bits() {
        let s = 20;
        let indep = rows(4000, s, |r| (0..s).map(|_| r.gen::<bool>() as u8).collect());
        let h = -ChowLiu::fit(&indep).log2_mode();
        assert!(h > 17.0 && h <= 20.5, "independent fair bits: {h}");
        let copied = rows(4000, s, |r| vec![r.gen::<bool>() as u8; s]);
        let h = -ChowLiu::fit(&copied).log2_mode();
        assert!(h < 1.5, "all bits equal: {h}");
    }

    #[test]
    fn monte_carlo_rank_puts_the_mode_first_and_the_rarest_string_last() {
        // skewed independent bits over 2^12 strings: all-zero is the mode, all-one the rarest
        let s = 12;
        let data = rows(20_000, s, |r| {
            (0..s).map(|_| (r.gen::<f64>() < 0.3) as u8).collect()
        });
        let ind = Indep::fit(&data);
        let zero = vec![vec![0u8; s]];
        let one = vec![vec![1u8; s]];
        let lr0 = mc_log2_ranks(
            &|x| ind.log2_prob(x),
            &|r, o| ind.sample(r, o),
            &zero,
            50_000,
            1,
        );
        let lr1 = mc_log2_ranks(
            &|x| ind.log2_prob(x),
            &|r, o| ind.sample(r, o),
            &one,
            200_000,
            1,
        );
        assert!(lr0[0] < 0.5, "the mode has rank 1: {}", lr0[0]);
        assert!(
            lr1[0] > 11.0,
            "the least likely string is near the end: {}",
            lr1[0]
        );
    }

    #[test]
    fn guesswork_bound_uses_the_tightest_point() {
        // T = 4 ranks 1, 2, 2^10, 2^10: point i=1 gives 0 + 2 = 2 bits
        let b = guesswork_bound(vec![0.0, 1.0, 10.0, 10.0]);
        assert!((b - 2.0).abs() < 1e-9, "{b}");
    }

    #[test]
    fn pair_unions_pick_the_largest_overlap_first() {
        let subsets = vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 2, 9],
            vec![10, 11, 12, 13],
        ];
        let u = pair_unions(&subsets, 4, 1, 2, 3);
        assert_eq!(u[0], vec![0, 1, 2, 3, 9]);
        assert_eq!(u.len(), 3);
    }
}
