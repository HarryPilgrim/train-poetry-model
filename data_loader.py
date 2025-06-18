import json

def extract_poems_from_json(json_path: str, output_txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    poems = []
    for poet in data:
        for poem in poet.get("poems", []):
            title = poem.get("title", "")
            content = poem.get("content", "")
            text = f"{title}\n{content}".strip()
            if len(text) > 30:
                poems.append(text)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(poems))