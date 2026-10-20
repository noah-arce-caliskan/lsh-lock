"""Protocol and native-result parsing tests; no NIST executable required."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location("nist_entropy", Path(__file__).with_name("nist_entropy.py"))
nist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nist)


def native_fixture():
    cases = []
    for i, name in enumerate(nist.TESTS):
        field = "tTupleRes" if name == "T-Tuple Test" else "lrsRes" if name == "LRS Test" else "hOriginal"
        cases.append({"testCaseDesc": name, field: .5 + i / 100})
    cases.append({"testCaseDesc": "Overall", "hAssessed": .5})
    return {"errorLevel": 0, "testCases": cases}


class ProtocolTests(unittest.TestCase):
    def test_refuses_existing_output_before_running_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(templates=Path(temp),output=Path(temp),trials=1,jobs=1,seed=67,timeout=1)
            with self.assertRaisesRegex(ValueError,"already exists"):
                nist.run(args)

    def test_native_failure_keeps_log_and_does_not_write_success_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir()
            for label in ['a','b']:
                (data/label).mkdir();(data/label/'r').write_text('0,1,0,1')
            binary=root/'fake_nist'
            binary.write_text('#!/bin/sh\nprintf "native failure\\n" >&2\nexit 9\n');binary.chmod(0o755)
            args=SimpleNamespace(templates=data,output=root/'out',trials=1,jobs=1,seed=67,timeout=15,
                nist_bin=str(binary),no_controls=True,keep_inputs=False)
            with self.assertRaisesRegex(RuntimeError,'exited 9'):
                nist.run(args)
            self.assertIn('native failure',(args.output/'raw/real_0000.log').read_text())
            self.assertFalse((args.output/'report.json').exists())

    def test_multiple_captures_and_variable_counts_keep_one_original_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for label, files in {"0101": ["z", "a"], "0019": ["b", "a", "c"]}.items():
                (root / label).mkdir()
                for i, name in enumerate(files):
                    (root / label / name).write_text(f" {i % 2}, 1,0,\n")
            bits, manifest, names, groups = nist.load_readings(root)
            self.assertEqual(names, ["0019", "0101"])
            self.assertEqual([len(g) for g in groups], [3, 2])
            self.assertEqual([m["reading"] for m in manifest], ["a", "b", "c", "a", "z"])
            np.testing.assert_array_equal(bits[0], [1, 1, 0])
            for seed in range(67, 100):
                selection = nist.select_rows(groups, seed)
                self.assertEqual([manifest[int(i)]["identity"] for i in selection], names)
                self.assertEqual(len(set(selection)), 2)
                np.testing.assert_array_equal(selection, nist.select_rows(groups, seed))

    def test_rejects_float_or_nonbinary_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root / "a").mkdir();(root / "a" / "reading").write_text("0,2,1")
            with self.assertRaisesRegex(ValueError, "binary"):
                nist.load_readings(root)

    def test_rejects_inconsistent_widths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for label, content in [("a", "0,1"), ("b", "0,1,0")]:
                (root / label).mkdir();(root / label / "reading").write_text(content)
            with self.assertRaisesRegex(ValueError, "length"):
                nist.load_readings(root)

    def test_controls_match_length_and_seed_without_using_real_bits(self):
        x=nist.control_bits(162,1280,67)
        self.assertEqual(x.shape,(162,1280))
        self.assertEqual(x.dtype,np.uint8)
        np.testing.assert_array_equal(x,nist.control_bits(162,1280,67))
        self.assertFalse(np.array_equal(x,nist.control_bits(162,1280,68)))


class NativeParsingTests(unittest.TestCase):
    def test_reads_all_ten_including_different_lrs_and_tuple_fields(self):
        minimum, tests=nist.parse_result(native_fixture())
        self.assertEqual(minimum,.5)
        self.assertEqual(tuple(tests),nist.TESTS)
        self.assertEqual(tests["LRS Test"],.55)

    def test_missing_estimator_is_not_silently_dropped(self):
        r=native_fixture();del r["testCases"][3]
        with self.assertRaisesRegex(ValueError,"Not all ten"):
            nist.parse_result(r)

    def test_nonfinite_or_missing_estimate_is_not_zero_filled(self):
        for value in [None,float("nan"),float("inf"),-1,1.01]:
            r=native_fixture();r["testCases"][0]["hOriginal"]=value
            with self.assertRaisesRegex(ValueError,"Missing/non-finite"):
                nist.parse_result(r)

    def test_rejects_native_error_and_inconsistent_overall(self):
        r=native_fixture();r["errorLevel"]=1
        with self.assertRaisesRegex(ValueError,"reported an error"):
            nist.parse_result(r)
        r=native_fixture();r["testCases"][-1]["hAssessed"]=.7
        with self.assertRaisesRegex(ValueError,"disagrees"):
            nist.parse_result(r)

    def test_rejects_duplicate_estimators(self):
        r=native_fixture();r["testCases"].append(copy.deepcopy(r["testCases"][0]))
        with self.assertRaisesRegex(ValueError,"Duplicate estimator"):
            nist.parse_result(r)

    def test_summary_does_not_mix_individual_test_minima(self):
        estimates={name:.9 for name in nist.TESTS}
        a={**estimates,nist.TESTS[0]:.4};b={**estimates,nist.TESTS[1]:.6}
        runs=[{"kind":"real","suite_minimum":min(v.values()),"estimates":v,
            "limiting_estimators":[min(v,key=v.get)]} for v in [a,b]]
        s=nist.summarize(runs)["real"]
        self.assertEqual(s["suite_minimum"]["median"],.5)
        self.assertEqual(s["per_estimator"][nist.TESTS[0]]["median"],.65)


if __name__ == "__main__":
    unittest.main()
