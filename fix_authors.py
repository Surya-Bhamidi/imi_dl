path = r'C:\Users\bhara\OneDrive\Desktop\IMI_DL 5\hybrid_materials_paper.tex'
content = open(path, 'r', encoding='utf-8').read()

old_author = r"""\author{
\IEEEauthorblockN{Bhamidi Naga Sai Surya Bharadwaj}
\IEEEauthorblockA{DL.AI.U4AID24012\\
\textit{Amrita School of AI}\\
\textit{Amrita Vishwa Vidyapeetham}, Faridabad, India}
\and
\IEEEauthorblockN{Alavala L H M Ramanjaneya Kumar}
\IEEEauthorblockA{DL.AI.U4AID24003\\
\textit{Amrita School of AI}\\
\textit{Amrita Vishwa Vidyapeetham}, Faridabad, India}
\and
\IEEEauthorblockN{Pilla Jasmitha}
\IEEEauthorblockA{DL.AI.U4AID24035\\
\textit{Amrita School of AI}\\
\textit{Amrita Vishwa Vidyapeetham}, Faridabad, India}
}"""

new_author = r"""\author{
\IEEEauthorblockN{Bhamidi Naga Sai Surya Bharadwaj\IEEEauthorrefmark{1},
Alavala L H M Ramanjaneya Kumar\IEEEauthorrefmark{1} and
Pilla Jasmitha\IEEEauthorrefmark{1}}
\IEEEauthorblockA{\IEEEauthorrefmark{1}Amrita School of AI,
Amrita Vishwa Vidyapeetham, Faridabad, India\\
Roll Nos: DL.AI.U4AID24012, DL.AI.U4AID24003, DL.AI.U4AID24035}
}"""

# Normalize line endings for matching
content_n = content.replace('\r\n', '\n')
old_n = old_author.replace('\r\n', '\n')

if old_n in content_n:
    content_n = content_n.replace(old_n, new_author)
    open(path, 'w', encoding='utf-8').write(content_n)
    print("Author block replaced successfully!")
else:
    print("ERROR: Could not find old author block")
    # Debug: show what's actually there
    idx = content_n.find(r'\author{')
    print(repr(content_n[idx:idx+400]))
