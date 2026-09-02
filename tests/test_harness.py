"""Master Test Harness and Reporter for BookTranslator E2E Test Suite.
Discovers and executes all 4 tiers of test suites, providing detailed timing, pass/fail matrices, and exit code 0 on full success.
"""
import os
import sys
import time
import argparse
import unittest
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root and venv site-packages are in sys.path
PROJECT_ROOT = Path.cwd().resolve()
for root_candidate in [Path.cwd().resolve(), Path(__file__).resolve().parent.parent, Path(os.getcwd()).resolve()]:
    if str(root_candidate) not in sys.path:
        sys.path.insert(0, str(root_candidate))
    venv_pkg = root_candidate / ".venv" / "Lib" / "site-packages"
    if venv_pkg.exists() and str(venv_pkg) not in sys.path:
        sys.path.insert(0, str(venv_pkg))



TIER_MAPPING = {
    1: ("Tier 1: Feature Coverage", "tests.e2e.tier1_feature_coverage"),
    2: ("Tier 2: Boundary & Corner Cases", "tests.e2e.tier2_boundary_corner"),
    3: ("Tier 3: Cross-Feature Combinations", "tests.e2e.tier3_cross_feature_pairwise"),
    4: ("Tier 4: Real-World Workloads", "tests.e2e.tier4_real_world_workloads"),
    5: ("Tier 5: Adversarial Hardening & Edge Cases", "tests.e2e.tier5_adversarial_hardening"),
}


class TestResultEntry:
    __test__ = False

    def __init__(self, tier: int, tier_name: str, module_name: str, test_name: str, status: str, duration_ms: float, error_msg: str = ""):
        self.tier = tier
        self.tier_name = tier_name
        self.module_name = module_name
        self.test_name = test_name
        self.status = status
        self.duration_ms = duration_ms
        self.error_msg = error_msg


class BookTranslatorTestRunner:
    def __init__(self, target_tiers: List[int] = None, filter_pattern: str = None, verbose: bool = True):
        self.target_tiers = target_tiers or [1, 2, 3, 4, 5]
        self.filter_pattern = filter_pattern.lower() if filter_pattern else None
        self.verbose = verbose
        self.results: List[TestResultEntry] = []

    def run(self) -> int:
        print("=" * 80)
        print("  BOOKTRANSLATOR OPAQUE-BOX E2E TEST HARNESS")
        print("  Architecture: 5-Tier Test Framework (Tiers 1-5)")
        print("=" * 80)

        suite_start = time.time()

        for tier_id in self.target_tiers:
            if tier_id not in TIER_MAPPING:
                print(f"[WARN] Unknown tier: {tier_id}")
                continue
            tier_name, package_name = TIER_MAPPING[tier_id]
            self._run_tier(tier_id, tier_name, package_name)

        total_duration = time.time() - suite_start
        return self._render_summary(total_duration)

    def _run_tier(self, tier_id: int, tier_name: str, package_name: str):
        print(f"\n>>> Executing {tier_name} ({package_name})")
        print("-" * 80)

        pkg_path = PROJECT_ROOT / package_name.replace(".", "/")
        if not pkg_path.exists():
            print(f"[ERROR] Directory not found: {pkg_path}")
            return

        py_files = sorted(pkg_path.glob("test_*.py"))
        if not py_files:
            print(f"[INFO] No test files found in {pkg_path}")
            return

        for py_file in py_files:
            mod_name = f"{package_name}.{py_file.stem}"
            if self.filter_pattern and self.filter_pattern not in mod_name.lower():
                continue
            self._run_module(tier_id, tier_name, mod_name, py_file)

    def _run_module(self, tier_id: int, tier_name: str, mod_name: str, file_path: Path):
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            tb = traceback.format_exc()
            self.results.append(TestResultEntry(
                tier=tier_id,
                tier_name=tier_name,
                module_name=file_path.name,
                test_name="<module_import>",
                status="FAIL",
                duration_ms=0.0,
                error_msg=f"Failed to import {mod_name}:\n{tb}"
            ))
            print(f"  [FAIL] {file_path.name}: Module Import Error")
            return

        # Find all test classes or test functions
        test_classes = []
        for attr_name in dir(mod):
            if attr_name.startswith("Test") and isinstance(getattr(mod, attr_name), type):
                cls = getattr(mod, attr_name)
                if getattr(cls, "__test__", True) is not False:
                    test_classes.append(cls)

        # If test classes exist
        if test_classes:
            for cls in test_classes:
                # Instantiate test class
                instance = cls()
                test_methods = [m for m in dir(cls) if m.startswith("test_") and callable(getattr(cls, m))]
                for method_name in test_methods:
                    if self.filter_pattern and self.filter_pattern not in method_name.lower():
                        continue
                    self._run_test_case(tier_id, tier_name, file_path.name, cls.__name__, method_name, getattr(instance, method_name))
        else:
            # Standalone test functions
            test_funcs = [getattr(mod, m) for m in dir(mod) if m.startswith("test_") and callable(getattr(mod, m))]
            for func in test_funcs:
                if self.filter_pattern and self.filter_pattern not in func.__name__.lower():
                    continue
                self._run_test_case(tier_id, tier_name, file_path.name, "<standalone>", func.__name__, func)

    def _run_test_case(self, tier_id: int, tier_name: str, file_name: str, class_name: str, method_name: str, func: Any):
        full_name = f"{class_name}.{method_name}" if class_name != "<standalone>" else method_name
        start_t = time.perf_counter()
        
        # Prepare parameters if fixture needed
        kwargs = {}
        import inspect
        import tempfile
        sig = inspect.signature(func)
        temp_dir_ctx = None
        
        if "temp_work_dir" in sig.parameters:
            temp_dir_ctx = tempfile.TemporaryDirectory()
            kwargs["temp_work_dir"] = Path(temp_dir_ctx.name)
            
        status = "PASS"
        error_msg = ""
        try:
            func(**kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            status = "FAIL"
            error_msg = traceback.format_exc()
        finally:
            if temp_dir_ctx is not None:
                try:
                    temp_dir_ctx.cleanup()
                except Exception:
                    pass
            duration_ms = (time.perf_counter() - start_t) * 1000.0

        self.results.append(TestResultEntry(
            tier=tier_id,
            tier_name=tier_name,
            module_name=file_name,
            test_name=full_name,
            status=status,
            duration_ms=duration_ms,
            error_msg=error_msg
        ))

        status_flag = "[PASS]" if status == "PASS" else "[FAIL]"
        if self.verbose:
            print(f"  {status_flag:<6} {file_name:<40} {full_name:<45} ({duration_ms:6.1f} ms)")
            if status == "FAIL":
                print(f"         Error: {error_msg.splitlines()[-1]}")

    def _render_summary(self, total_duration: float) -> int:
        print("\n" + "=" * 80)
        print("  E2E TEST EXECUTION SUMMARY")
        print("=" * 80)

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.status == "PASS")
        failed_tests = sum(1 for r in self.results if r.status == "FAIL")

        print(f"{'Tier':<35} | {'Passed':<8} | {'Failed':<8} | {'Total':<8} | {'Time (ms)'}")
        print("-" * 80)

        for tier_id in self.target_tiers:
            tier_name, _ = TIER_MAPPING[tier_id]
            tier_results = [r for r in self.results if r.tier == tier_id]
            t_pass = sum(1 for r in tier_results if r.status == "PASS")
            t_fail = sum(1 for r in tier_results if r.status == "FAIL")
            t_tot = len(tier_results)
            t_time = sum(r.duration_ms for r in tier_results)
            print(f"{tier_name:<35} | {t_pass:<8} | {t_fail:<8} | {t_tot:<8} | {t_time:8.1f} ms")

        print("-" * 80)
        print(f"{'TOTAL':<35} | {passed_tests:<8} | {failed_tests:<8} | {total_tests:<8} | {total_duration*1000:8.1f} ms")
        print("=" * 80)

        if failed_tests > 0:
            print("\n[FAILED TESTS DETAILS]")
            for r in self.results:
                if r.status == "FAIL":
                    print(f"\n--- {r.module_name} :: {r.test_name} ---")
                    print(r.error_msg)
            print("\n>>> TEST SUITE RESULT: FAILED (Exit Code 1)\n")
            return 1
        else:
            print(f"\n>>> TEST SUITE RESULT: ALL {total_tests} TESTS PASSED (Exit Code 0)\n")
            return 0


def main():
    parser = argparse.ArgumentParser(description="BookTranslator E2E Master Test Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific test tier only")
    parser.add_argument("--filter", type=str, help="Filter tests by name pattern")
    parser.add_argument("--quiet", action="store_true", help="Minimize console output")
    args = parser.parse_args()

    target_tiers = [args.tier] if args.tier else [1, 2, 3, 4, 5]
    runner = BookTranslatorTestRunner(
        target_tiers=target_tiers,
        filter_pattern=args.filter,
        verbose=not args.quiet
    )
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
