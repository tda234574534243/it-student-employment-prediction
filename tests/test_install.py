import install


def test_install_dataset_downloads_and_copies_files(tmp_path, monkeypatch, capsys):
    source = tmp_path / "download"
    destination = tmp_path / "installed"
    nested_source = source / "Student Placement Dataset"
    nested_source.mkdir(parents=True)
    (source / "README.txt").write_text("dataset", encoding="utf-8")
    (nested_source / "train.csv").write_text("id,status\n1,Placed\n", encoding="utf-8")

    calls = []

    def fake_dataset_download(handle):
        calls.append(handle)
        return str(source)

    monkeypatch.setattr(install.kagglehub, "dataset_download", fake_dataset_download)
    monkeypatch.setattr(install, "LOCAL_PATH", str(destination))

    install.install_dataset()

    assert calls == [install.DATASET_HANDLE]
    assert (destination / "README.txt").read_text(encoding="utf-8") == "dataset"
    assert (
        destination / "Student Placement Dataset" / "train.csv"
    ).read_text(encoding="utf-8") == "id,status\n1,Placed\n"
    assert capsys.readouterr().out == f"Path to dataset files: {destination}\n"


def test_copy_dataset_contents_merges_existing_directories(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source_subdir = source / "nested"
    destination_subdir = destination / "nested"
    source_subdir.mkdir(parents=True)
    destination_subdir.mkdir(parents=True)
    (source_subdir / "replace.txt").write_text("new", encoding="utf-8")
    (destination_subdir / "replace.txt").write_text("old", encoding="utf-8")
    (destination_subdir / "keep.txt").write_text("existing", encoding="utf-8")

    install.copy_dataset_contents(str(source), str(destination))

    assert (destination_subdir / "replace.txt").read_text(encoding="utf-8") == "new"
    assert (destination_subdir / "keep.txt").read_text(encoding="utf-8") == "existing"
