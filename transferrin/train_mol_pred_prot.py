from eval_tasks.trainer import main as trainer_task_main

"use_fuse, use_model, bs, lr, drop_out, hidden_dim, task_name, fuse_base, mol_emd, protein_emd, n_layers,
metric, max_no_improve, fuse_model = None, task_suffix
""
trainer_task_main(use_fuse=True, use_model=False, bs=32, lr=1e-4, drop_out=0.1, hidden_dim=128, task_name="BBBP",
                  fuse_base=""
                  )
