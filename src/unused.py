@tool
def get_files_names() -> str:
    """
    Lista todos os arquivos disponíveis na pasta documentos em ordem alfabética.
    Use esta ferramenta SEMPRE PRIMEIRO antes de qualquer consulta sobre regulamentação.
    """
    try:
        base_dir = Path(__file__).parent.parent / "documentos"
        if not base_dir.exists():
            return "Pasta documentos não encontrada."
            
        files = sorted([f.name for f in base_dir.iterdir() if f.is_file()])
        
        if not files:
            return "Nenhum arquivo encontrado na pasta documentos."
        
        return f"Arquivos disponíveis: {', '.join(files)}"
        
    except Exception as e:
        return f"Erro ao listar arquivos: {str(e)}"


@tool
def pdf_converter(filename: str) -> str:
    """
    FERRAMENTA OBRIGATÓRIA para consultar documentos universitários.
    Use para qualquer pergunta sobre regulamentos ou vida acadêmica.
    Parâmetro: nome do arquivo (ex: 'regulamento', 'regulamento.pdf', 'regulamento.json')
    A ferramenta automaticamente processa PDFs e JSONs.
    """
    try:
        base_dir = Path(__file__).parent.parent / "documentos"
        
        # Processa o nome do arquivo
        filename = filename.strip()
        if filename.endswith(".json"):
            json_path = base_dir / filename
            pdf_path = json_path.with_suffix(".pdf")
        elif filename.endswith(".pdf"):
            pdf_path = base_dir / filename
            json_path = pdf_path.with_suffix(".json")
        else:
            pdf_path = base_dir / f"{filename}.pdf"
            json_path = base_dir / f"{filename}.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        else:
            if not pdf_path.exists():
                available_files = [f.name for f in base_dir.glob("*.pdf")]
                return f"PDF não encontrado: {pdf_path.name}. Arquivos PDF disponíveis: {', '.join(available_files) if available_files else 'Nenhum'}"
            
            # Converte o PDF
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            document = result.document
            json_data = document.export_to_dict()
            
            # Salva o JSON (se conseguir)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Continua mesmo se não conseguir salvar
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"