"""Public source export includes new modules and omits local sharing overlays."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def test_source_export_preserves_local_build_requirements_without_private_overlays(tmp_path,monkeypatch):
    source=tmp_path/'source'
    source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    (source/'.gitignore').write_text('.local/\n.agents/\nbuild/\n*.egg-info/\n')
    (source/'tracked.py').write_text('tracked=1\n')
    subprocess.run(['git','-C',str(source),'add','.gitignore','tracked.py'],check=True)
    for name,body in {
        'march/new_module.py':'new_module=2\n',
        'third_party/vllm/requirements/build/cuda.txt':'ninja\n',
        '.local/checkpoints.env':'private fixture\n',
        '.agents/private.txt':'private fixture\n',
        'build/generated.py':'generated fixture\n',
    }.items():
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(body)
    subprocess.run(['git','-C',str(source),'add','-f','third_party/vllm/requirements/build/cuda.txt'],check=True)
    scripts=source/'scripts/march'; scripts.mkdir(parents=True)
    shutil.copyfile(ROOT/'scripts/march/check_public_release.py',scripts/'check_public_release.py')
    spec=importlib.util.spec_from_file_location('public_source_export_fixture',ROOT/'scripts/march/export_source.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module,'__file__',str(scripts/'export_source.py'))
    output=tmp_path/'source.tar.gz'
    monkeypatch.setattr(sys,'argv',['export_source.py','--output',str(output)])
    module.main()
    with tarfile.open(output) as archive:
        names=archive.getnames()
    assert 'march-safin1/march/new_module.py' in names
    assert 'march-safin1/third_party/vllm/requirements/build/cuda.txt' in names
    assert not any(len(Path(name).parts)>1 and Path(name).parts[1] in {'.git','.local','.agents','build'}
                   for name in names)
    assert output.with_suffix('.gz.sha256').is_file()
