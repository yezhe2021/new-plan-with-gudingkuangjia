"""Export the source fixed KV using Qwen and the public header only."""
from types import SimpleNamespace
from experiment_common import *

@torch.no_grad()
def main():
    verify()
    if done('export/completed.json'):
        assert C.sha(ROOT/'export/canonical.pt')==C.read(ROOT/'export/completed.json')['file_sha256'];return
    tok,model=C.load_model(S.C.CFG['models']['receiver'],'cuda',dtype=torch.float16)
    prefix=C.read(S.C.ROOT/'manifest.json')['prefix_ids']
    header=C.decoder(model)(torch.tensor([prefix],device='cuda'),use_cache=True).past_key_values
    engine=SimpleNamespace(model=model,prefix_ids=prefix,header=header)
    module=load_static(engine);before=C.versions(module);values=module.canonical()
    assert len(values)==72 and all(x.shape==(1,8,32,128) for x in values)
    assert C.versions(module)==before and all(not p.requires_grad and p.grad is None for p in module.parameters())
    path=ROOT/'export/canonical.pt';path.parent.mkdir(exist_ok=True)
    C.atomic_save(dict(manifest_sha256=identity(),source_checkpoint_sha256=CFG['static_checkpoint_sha256'],prefix_ids=prefix,receiver_config_sha256=C.digest(model.config.to_dict()),canonical_memory=tuple(x.cpu() for x in values),header_values=tuple(x.cpu() for x in K.flat(header))),path)
    mark('export/completed.json',status='completed',source_checkpoint_sha256=CFG['static_checkpoint_sha256'],file_sha256=C.sha(path),optimizer_updates=0,public_header_only=True,bytes=path.stat().st_size)
    module.routing.close();C.log('SOURCE STATIC KV EXPORTED WITHOUT SENDER',path.stat().st_size)

if __name__=='__main__':main()
