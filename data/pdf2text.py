from pdfminer.high_level import extract_text
import os
from tqdm import tqdm

pdf_path = 'docs'
files = ['1.pdf', '2.pdf', '3.pdf', '4.pdf', '5.pdf']

for file in tqdm(files):
    text = extract_text(os.path.join(pdf_path, file))
    file_name = os.path.splitext(file)[0]
    with open(f'docs/{file_name}.txt', 'w', encoding='utf-8') as f:
        f.write(text)        