path = r'C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\hybrid_materials_paper.tex'
content = open(path, 'r', encoding='utf-8').read()

# Fix the duplicate begin
content = content.replace(r'\begin{figure*}\begin{figure*}[t]', r'\begin{figure*}[t]')

# Fix the corrupted end
content = content.replace(r'\end{figure*}NN Backbone}', r'\end{figure*}' + '\n\n' + r'\subsection{DNN Backbone}')

open(path, 'w', encoding='utf-8').write(content)
print("Fixed!")
# Verify
lines = content.split('\n')
for i, l in enumerate(lines[110:185], start=111):
    if 'figure*' in l or 'Backbone' in l:
        print(f"  Line {i}: {l.strip()}")
