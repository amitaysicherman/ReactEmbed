import numpy as np
import torch
from tqdm import tqdm

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.models import LinFuseModel
from eval_tasks.trainer import main as trainer_task_main
from preprocessing.seq_to_vec import SeqToVec
from transferrin.utils import get_human_enzyme_binding_proteins, get_all_sequences

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p_model = "esm3-medium"
m_model = "ChemBERTa"
fuse_base = "data/reactome/model/esm3-medium-ChemBERTa-2-256-0.3-10-0.001-8192-0.0"
model: LinFuseModel = None
score, model = trainer_task_main(use_fuse=True, use_model=False, bs=32, lr=1e-4, drop_out=0.1, hidden_dim=128,
                                 task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                 n_layers=2, metric="auc", max_no_improve=15, return_model=True)

all_seq = get_all_sequences()
human_enzyme_binding_proteins = get_human_enzyme_binding_proteins()
vecs = []
names = []
seq_to_vec = SeqToVec(model_name=p_model)

for i, seq in tqdm(enumerate(all_seq)):
    vec = seq_to_vec.to_vec(seq)
    if vec is not None:
        vecs.append(vec)
        names.append(human_enzyme_binding_proteins[i])

proteins = torch.tensor(vecs).to(device)
fuse_model: ReactEmbedModel = model.fuse_model
fuse_model.eval()
x = fuse_model.dual_forward(proteins, "P")
pred = model.layers(x)
res = torch.sigmoid(pred).detach().cpu().numpy()
# get top 10 indexes
top_10 = np.argsort(res.flatten())[-10:]
print(top_10)
with open("data/reactome/proteins.txt") as f:
    lines = f.readlines()
for i in top_10:
    print(lines[i])

transferrin_seq = "MRLAVGALLVCAVLGLCLAVPDKTVRWCAVSEHEATKCQSFRDHMKSVIPSDGPSVACVKKASYLDCIRAIAANEADAVTLDAGLVYDAYLAPNNLKPVVAEFYGSKEDPQTFYYAVAVVKKDSGFQMNQLRGKKSCHTGLGRSAGWNIPIGLLYCDLPEPRKPLEKAVANFFSGSCAPCADGTDFPQLCQLCPGCGCSTLNQYFGYSGAFKCLKDGAGDVAFVKHSTIFENLANKADRDQYELLCLDNTRKPVDEYKDCHLAQVPSHTVVARSMGGKEDLIWELLNQAQEHFGKDKSKEFQLFSSPHGKDLLFKDSAHGFLKVPPRMDAKMYLGYEYVTAIRNLREGTCPEAPTDECKPVKWCALSHHERLKCDEWSVNSVGKIECVSAETTEDCIAKIMNGEADAMSLDGGFVYIAGKCGLVPVLAENYNKSDNCEDTPEAGYFAIAVVKKSASDLTWDNLKGKKSCHTAVGRTAGWNIPMGLLYNKINHCRFDEFFSEGCAPGSKKDSSLCKLCMGSGLNLCEPNNKEGYYGYTGAFRCLVEKGDVAFVKHQTVPQNTGGKNPDPWAKNLNEKDYELLCLDGTRKPVEEYANCHLARAPNHAVVTRKDKEACVHKILRQQQHLFGSNVTDCSGNFCLFRSETKDLLFRDDTVCLAKLHDRNTYEKYLGEEYVKAVGNLRKCSTSSLLEACTFRRP"
transferrin_vec = seq_to_vec.to_vec(transferrin_seq)
transferrin_vec = torch.Tensor(transferrin_vec).to(device)
transferrin_vec = transferrin_vec.unsqueeze(0)
t_x = fuse_model.dual_forward(transferrin_vec, "P")
t_pred = model.layers(t_x)
t_res = torch.sigmoid(t_pred).detach().cpu().numpy()
print(f"Transferrin score: {t_res}")

transferrin_rank = np.sum(res.flatten() < t_res.flatten()[0])

print(f"Transferrin rank: {transferrin_rank}")

import matplotlib.pyplot as plt

plt.hist(res.flatten(), bins=100)
# add a line for transferrin
plt.axvline(t_res, color='r', linestyle='dashed', linewidth=1)
plt.show()
