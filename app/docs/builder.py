from docxtpl import DocxTemplate
from datetime import date
from pathlib import Path

def build_doc(context: dict, template_path: str, out_path: str) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tpl = DocxTemplate(template_path)
    tpl.render({**context, "today": date.today().isoformat()})
    tpl.save(out_path)
    return out_path
