"""Regression checks for the actual demo script and npm release validator.

Run from the repository root: python3 -m unittest discover -s scripts/tests -v
Requires Node.js; does not download models, publish packages, or run cargo builds.
"""

import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ("darwin-arm64", "darwin-x64", "linux-arm64-gnu", "linux-x64-gnu")


class ReleaseVersions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "crates/memme-node"
        for folder in (".", *(f"npm/{p}" for p in PLATFORMS)):
            dest = self.base / folder
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "crates/memme-node" / folder / "package.json", dest / "package.json")
        self.version = json.loads((self.base / "package.json").read_text())["version"]
        workflow = (ROOT / ".github/workflows/publish-node.yml").read_text()
        # Execute the inline validator shipped in the workflow, including for
        # old tags whose source does not contain a separate validation script.
        match = re.search(r"node <<'JS'\n(.*?)^          JS$", workflow, re.M | re.S)
        self.assertIsNotNone(match)
        self.validator = textwrap.dedent(match.group(1))

    def validate(self, tag=None):
        return subprocess.run(
            ["node", "-e", self.validator], cwd=self.root,
            env={**os.environ, "RELEASE_TAG": tag or f"v{self.version}"},
            capture_output=True, text=True, timeout=10,
        )

    def change(self, folder, field, value):
        path = self.base / folder / "package.json"
        data = json.loads(path.read_text())
        data[field] = value
        path.write_text(json.dumps(data))

    def test_consistent_release_passes(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_tag_cannot_publish_new_branch_metadata(self):
        self.change(".", "version", "99.0.0")
        result = self.validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Main package differs from release tag", result.stderr)

    def test_tag_must_match_package_version(self):
        self.assertNotEqual(self.validate("v99.0.0").returncode, 0)

    def test_each_platform_version_is_checked(self):
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                self.change(f"npm/{platform}", "version", "99.0.0")
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{platform} differs from release tag", result.stderr)
                self.change(f"npm/{platform}", "version", self.version)

    def test_each_optional_dependency_is_checked(self):
        dependencies = json.loads((self.base / "package.json").read_text())["optionalDependencies"]
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                changed = {**dependencies, f"memme-{platform}": "99.0.0"}
                self.change(".", "optionalDependencies", changed)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{platform} dependency differs", result.stderr)
        self.change(".", "optionalDependencies", dependencies)


class DemoProcess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"

    def run_demo(self, script, port, extra_env=None):
        env = {**os.environ, "MEMME_DEMO_DIR": str(self.data),
               "MEMME_DEMO_PORT": str(port), "MEMME_DEMO_START_TIMEOUT": "600"}
        env.pop("CARGO_BUILD_TARGET", None)
        env.update(extra_env or {})
        return subprocess.run(["bash", str(script)], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=10)

    def test_occupied_port_is_rejected_without_contacting_existing_service(self):
        with socket.socket() as existing:
            existing.bind(("127.0.0.1", 0))
            existing.listen()
            result = self.run_demo(ROOT / "demos/rest-demo.sh", existing.getsockname()[1])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cannot use demo port", result.stderr)
            self.assertNotIn("PASS", result.stdout)
            self.assertFalse(self.data.exists())
            self.assertEqual(select.select([existing], [], [], 0)[0], [])

    def test_child_startup_failure_is_reported_without_waiting_for_timeout(self):
        # Run the real script in a tiny fixture project. Only native downloads
        # and cargo are substituted; the child exits as a failed model init would.
        for folder in ("demos", "scripts", "bin", "target/release"):
            (self.root / folder).mkdir(parents=True)
        script = self.root / "demos/rest-demo.sh"
        shutil.copyfile(ROOT / "demos/rest-demo.sh", script)
        for name in ("download-vexdb-lite-extension.sh", "download-onnx-runtime.sh"):
            (self.root / "scripts" / name).write_text("printf '%s\\n' /fixture/library\n")
        cargo = self.root / "bin/cargo"
        cargo.write_text("#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
                         "if sys.argv[1] == 'metadata':\n"
                         " print(json.dumps({'target_directory': str(Path(__file__).resolve().parents[1] / 'target')}))\n")
        cargo.chmod(0o700)
        server = self.root / "target/release/memme-server"
        server.write_text("#!/usr/bin/env bash\necho 'fixture initialization failed' >&2\nexit 27\n")
        server.chmod(0o700)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        result = self.run_demo(script, port, {"PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Demo server exited during startup", result.stderr)
        self.assertIn("fixture initialization failed", result.stderr)
        self.assertNotIn("PASS", result.stdout)
        self.assertFalse((self.data / "demo-memory.db").exists())


if __name__ == "__main__":
    unittest.main()
