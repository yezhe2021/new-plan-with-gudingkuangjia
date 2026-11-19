"""Run an unchanged original phase with a separately pinned cache extension."""
import json
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent;TOP=HERE.parent
sys.path.insert(0,str(TOP))
import gemma_common as H
manifest=json.loads((HERE/'cache_manifest.json').read_text())
assert H.identity()==manifest['parent_manifest_sha256']
for name,digest in manifest['code'].items():assert H.C.sha(HERE/name)==digest,name
sys.path.insert(0,str(HERE))
from latent_cache import install
script=sys.argv[1];assert script in ['cpu_checks.py','smoke.py','train_gemma.py','evaluate_gemma.py']
install(H,manifest['contract'],script)
sys.argv=[str(TOP/script),*sys.argv[2:]]
runpy.run_path(str(TOP/script),run_name='__main__')
