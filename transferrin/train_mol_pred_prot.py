import torch

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.models import LinFuseModel
from eval_tasks.trainer import main as trainer_task_main
from preprocessing.seq_to_vec import SeqToVec
from transferrin.utils import get_go_terms, get_vecs, find_optimal_filter_columns, get_go_matrix, \
    get_go_ancestors_cached

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p_model = "esm3-medium"
m_model = "ChemBERTa"
fuse_base = "data/reactome/model/esm3-medium-ChemBERTa-2-256-0.3-10-0.001-8192-0.0"
model: LinFuseModel = None
score, model = trainer_task_main(use_fuse=True, use_model=False, bs=32, lr=1e-4, drop_out=0.1, hidden_dim=128,
                                 task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                 n_layers=2, metric="auc", max_no_improve=15, return_model=True)

vecs = get_vecs()
proteins = torch.tensor(vecs).to(device)
fuse_model: ReactEmbedModel = model.fuse_model
fuse_model.eval()
x = fuse_model.dual_forward(proteins, "P")
pred = model.layers(x)
res = torch.sigmoid(pred).detach().cpu().numpy()

transferrin_id = "P02787"
transferrin_seq = "MRLAVGALLVCAVLGLCLAVPDKTVRWCAVSEHEATKCQSFRDHMKSVIPSDGPSVACVKKASYLDCIRAIAANEADAVTLDAGLVYDAYLAPNNLKPVVAEFYGSKEDPQTFYYAVAVVKKDSGFQMNQLRGKKSCHTGLGRSAGWNIPIGLLYCDLPEPRKPLEKAVANFFSGSCAPCADGTDFPQLCQLCPGCGCSTLNQYFGYSGAFKCLKDGAGDVAFVKHSTIFENLANKADRDQYELLCLDNTRKPVDEYKDCHLAQVPSHTVVARSMGGKEDLIWELLNQAQEHFGKDKSKEFQLFSSPHGKDLLFKDSAHGFLKVPPRMDAKMYLGYEYVTAIRNLREGTCPEAPTDECKPVKWCALSHHERLKCDEWSVNSVGKIECVSAETTEDCIAKIMNGEADAMSLDGGFVYIAGKCGLVPVLAENYNKSDNCEDTPEAGYFAIAVVKKSASDLTWDNLKGKKSCHTAVGRTAGWNIPMGLLYNKINHCRFDEFFSEGCAPGSKKDSSLCKLCMGSGLNLCEPNNKEGYYGYTGAFRCLVEKGDVAFVKHQTVPQNTGGKNPDPWAKNLNEKDYELLCLDGTRKPVEEYANCHLARAPNHAVVTRKDKEACVHKILRQQQHLFGSNVTDCSGNFCLFRSETKDLLFRDDTVCLAKLHDRNTYEKYLGEEYVKAVGNLRKCSTSSLLEACTFRRP"
seq_to_vec = SeqToVec(p_model)
transferrin_vec = seq_to_vec.to_vec(transferrin_seq)
transferrin_vec = torch.Tensor(transferrin_vec).to(device)
transferrin_vec = transferrin_vec.unsqueeze(0)
t_x = fuse_model.dual_forward(transferrin_vec, "P")
t_pred = model.layers(t_x)
t_res = torch.sigmoid(t_pred).detach().cpu().numpy()

t_go_terms = list(get_go_terms(transferrin_id))
t_go_terms_with_ans = sum([get_go_ancestors_cached(go_term) for go_term in t_go_terms], [])
t_go_terms.extend(t_go_terms_with_ans)
t_go_terms = list(set(t_go_terms))

go_matrix = get_go_matrix()
go_matrix["score"] = res.flatten()

go_matrix.loc[transferrin_id] = 0
go_matrix.loc[transferrin_id, t_go_terms] = 1
go_matrix.loc[transferrin_id, "score"] = t_res.flatten()[0]
transferrin_index = go_matrix.index.get_loc(transferrin_id)
go_matrix['R'] = go_matrix['score'].rank(ascending=False, method='min')
del go_matrix['score']
print(find_optimal_filter_columns(go_matrix, transferrin_index, 500, n=3))
