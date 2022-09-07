import filecmp
import os
import shutil
from pathlib import Path

from stracker import Asset, ExperimentAssetsManager, ProjectAssetsManager
from tqdm import tqdm


class testAsset:
    def __init__(self):
        self.root = Path("/Volumes/DATA/test/asset")

        self.test_init()
        print("Pass init test.")

        self.test_read()
        print("Pass read test.")

    def test_init(self):
        for i, f in enumerate(self.root.glob("**/*")):
            if f.is_dir():
                continue
            asset = Asset(i, f, self.root)

    def test_read(self):
        for i, f in enumerate(self.root.glob("**/*")):
            if f.is_dir():
                continue
            asset = Asset(i, f, self.root)
            d = asset.read()


class testExp:
    def __init__(self):
        self.root = Path("/Volumes/DATA/test/exp")

        self.test_init_update()
        print("Pass init update test.")

        self.test_add_get_delete()
        print("Pass add get delete test.")

        self.test_compress_uncompress()
        print("Pass compress uncompress test.")

    def _update_consistent(self, exp):
        assert {str(f.absolute()) for f in exp.path.glob("**/*") if f.is_file()} == {
            str(a.path.absolute()) for a in exp._assets.values()
        }

        assert {
            str(f.relative_to(exp.path)) for f in exp.path.glob("**/*") if f.is_file()
        } == set(exp._assets.keys())

    def test_init_update(self):
        for d in self.root.glob("*"):
            exp = ExperimentAssetsManager(d)
            assert exp.key == d.stem
            self._update_consistent(exp)
        print(exp)

    def test_add_get_delete(self):
        test_name = "test_exp_add.txt"
        with open(test_name, "wb") as f:
            f.write(os.urandom(2048))
        for d in self.root.glob("*"):
            exp = ExperimentAssetsManager(d)
            exp.add(test_name)
            self._update_consistent(exp)

            a = exp.get(test_name)
            assert a is not None
            self._update_consistent(exp)
            assert filecmp.cmp(test_name, str(a.path.absolute()))

            exp.remove_asset(test_name)
            self._update_consistent(exp)
            assert test_name not in exp.asset_paths
            assert not (exp.path / test_name).is_file()

            dest_name = f"no_exist/not_exist/{test_name}"
            exp.add(test_name, dest_name)
            self._update_consistent(exp)

            a = exp.get(dest_name)
            assert a is not None
            self._update_consistent(exp)
            assert filecmp.cmp(test_name, str(a.path.absolute()))

            exp.remove_asset(dest_name)
            self._update_consistent(exp)
            assert dest_name not in exp.asset_paths
            assert not (exp.path / dest_name).is_file()

    def _compress_basic(self, exp):
        assert exp.path.is_file()
        assert exp.is_compressed
        ppath = self.root / exp.path.stem
        assert not ppath.is_file() and not ppath.is_dir()
        assert exp.path.suffix == ".zip"
        assert len(exp) == 0
        assert len(exp._assets) == 0
        assert len(exp._key_asset_mapping) == 0

    def _uncompress_basic(self, exp):
        assert exp.path.is_dir()
        assert not exp.is_compressed
        ppath = self.root / f"{exp.path.stem}.zip"
        assert not ppath.is_file() and not ppath.is_dir()
        assert exp.path.suffix == ""

    def test_compress_uncompress(self):
        for d in self.root.glob("*"):
            exp = ExperimentAssetsManager(d)
            exp.compress()
            self._compress_basic(exp)

        assert all([f.suffix == ".zip" for f in self.root.glob("*")])
        print(exp)

        pro_root = Path("/Volumes/DATA/test/pro")
        for d in self.root.glob("*"):
            exp = ExperimentAssetsManager(d)

            if exp.path.stem not in [
                "0ad41f66-9e1c-47d5-baea-1b71c9a30828",
                "0ae6309c-8586-4f95-b5fc-58c373968f28",
            ]:
                self._compress_basic(exp)
                exp.uncompress()
            self._uncompress_basic(exp)
            exp_orig = ExperimentAssetsManager(pro_root / exp.path.name)
            self._uncompress_basic(exp_orig)

            assert set(exp.asset_paths) == set(exp_orig.asset_paths)
            for a in exp._assets.values():
                a1 = exp_orig.get(path=a.rel_path)
                filecmp.cmp(str(a.path.absolute()), str(a1.path.absolute()))

        print(exp)


class testPro:
    def __init__(self):
        self.root = Path("/Volumes/DATA/test/pro")
        # pro = ProjectAssetsManager(self.root)

        # for k in pro.keys[:5]:
        #     exp = pro.get(k)
        #     exp.compress()

        self.test_init_update()
        print("Pass init update test.")

        # self.test_get_remove()
        # print("Pass get remove test.")

        self.test_compress_uncompress()
        print("Pass compress uncompress test.")

    def _update_consistency(self, pro):
        k1 = [d.stem for d in self.root.glob("*") if d.is_dir() or d.suffix == ".zip"]
        assert set(k1) == set(pro.keys)

    def test_init_update(self):
        pro = ProjectAssetsManager(self.root)
        self._update_consistency(pro)
        print(pro)

    def test_get_remove(self):
        exp_root = Path("/Volumes/DATA/test/exp")
        pro = ProjectAssetsManager(self.root)
        for k in tqdm(pro.keys):
            exp1 = pro.get(k)
            exp2 = pro.get(path=pro.path / k)
            exp3 = pro.get(path=pro.path / f"{k}.zip")
            assert exp1.path == exp2.path
            assert exp1.path == exp3.path
            assert exp1.path in [pro.path / k, pro.path / f"{k}.zip"]

            pro.remove_exp(key=k)
            self._update_consistency(pro)
            assert k not in pro.keys
            assert pro.get(k) is None

            shutil.copytree(
                str((exp_root / k).absolute()), str(self.root.absolute() / k)
            )
            pro.update()
            self._update_consistency(pro)
            assert k in pro.keys
            assert pro.get(k) is not None

            pro.remove_exp(path=self.root / k)
            self._update_consistency(pro)
            assert k not in pro.keys
            assert pro.get(k) is None

            shutil.copytree(
                str((exp_root / k).absolute()), str(self.root.absolute() / k)
            )
            pro.update()
            self._update_consistency(pro)
            assert k in pro.keys
            assert pro.get(k) is not None

    def _compress_basic(self, pro, remove_old):
        assert pro.path.is_file()
        assert pro.is_compressed
        ppath = self.root
        if remove_old:
            assert not ppath.is_file() and not ppath.is_dir()
        else:
            assert ppath.is_dir()
        assert pro.path.suffix == ".zip"
        assert len(pro) == 0

    def _uncompress_basic(self, pro, remove_old):
        assert pro.path.is_dir()
        assert not pro.is_compressed
        ppath = self.root.with_suffix(".zip")
        if remove_old:
            assert not ppath.is_file() and not ppath.is_dir()
        else:
            assert ppath.is_file()
        assert pro.path.suffix == ""

    def test_compress_uncompress(self):
        exp_root = Path("/Volumes/DATA/test/exp")
        pro_orig = ProjectAssetsManager(exp_root)
        pro = ProjectAssetsManager(self.root)
        self._uncompress_basic(pro, True)

        pro.compress(True)
        self._compress_basic(pro, True)
        print(pro)

        pro.uncompress(True)
        self._uncompress_basic(pro, True)
        print(pro)

        assert set(pro.keys) == set(pro_orig.keys)
        for k in pro.keys:
            exp1 = pro.get(k)
            exp2 = pro.get(k)
            for a in exp1.assets:
                a1 = exp2.get(path=a.rel_path)
                filecmp.cmp(str(a.path.absolute()), str(a1.path.absolute()))

        pro.compress(False)
        self._compress_basic(pro, False)
        print(pro)
        shutil.rmtree(self.root)

        pro.uncompress(False)
        self._uncompress_basic(pro, False)
        print(pro)

        assert set(pro.keys) == set(pro_orig.keys)
        for k in pro.keys:
            exp1 = pro.get(k)
            exp2 = pro.get(k)
            for a in exp1.assets:
                a1 = exp2.get(path=a.rel_path)
                filecmp.cmp(str(a.path.absolute()), str(a1.path.absolute()))


# testAsset()
# testExp()
testPro()
