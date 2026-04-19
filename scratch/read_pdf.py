import fitz

doc = fitz.open(r"d:\AI Agent for Hackathon\shopwave-agent\Hackathon_2026.pdf")
text = ""
for i, page in enumerate(doc):
    text += f"=== PAGE {i+1} ===\n"
    text += page.get_text() + "\n\n"

with open(r"d:\AI Agent for Hackathon\shopwave-agent\pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("PDF converted successfully!")
