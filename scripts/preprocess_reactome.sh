SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$SCRIPT_DIR"
pip install -e .

wget https://reactome.org/download/current/biopax.zip -O data/biopax/biopax.zip
unzip data/biopax/biopax.zip -j Homo_sapiens.owl -d data/biopax
rm data/biopax/biopax.zip

echo "Start biopax_parser"
python preprocessing/biopax_parser.py

echo "Start build_index"
python preprocessing/build_index.py

echo "Start seq_api"
python preprocessing/seq_api.py

echo "Start seq_to_vec"
python preprocessing/seq_to_vec.py