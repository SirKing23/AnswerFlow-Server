import io
import openpyxl


def parse_xlsx(file_bytes: bytes) -> str:
    """
    Parse .xlsx files using openpyxl in read_only mode.

    read_only=True is critical — without it openpyxl loads the entire
    workbook object into memory. A 10MB xlsx can expand to 100MB+ in memory.
    read_only streams row by row instead.

    Strategy: convert each row into "Header: Value | Header: Value" format
    so each row embeds with full context. This is far better than embedding
    raw CSV-style text where column meaning is lost.
    """
    wb = openpyxl.load_workbook(
        io.BytesIO(file_bytes),
        read_only=True,   # streaming mode — critical for memory
        data_only=True    # get computed values, not formulas
    )

    all_sheets_text = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        # First row is headers
        headers = [str(cell).strip() if cell is not None else f"Column_{i}"
                   for i, cell in enumerate(rows[0])]

        sheet_rows = []
        for row in rows[1:]:
            # Skip completely empty rows
            if all(cell is None for cell in row):
                continue

            # Build "Header: Value | Header: Value" per row
            readable = " | ".join(
                f"{headers[i]}: {str(cell).strip()}"
                for i, cell in enumerate(row)
                if i < len(headers) and cell is not None and str(cell).strip()
            )
            if readable:
                # Prepend sheet name for context
                sheet_rows.append(f"[Sheet: {sheet_name}] {readable}")

        if sheet_rows:
            all_sheets_text.append("\n".join(sheet_rows))

    wb.close()

    return "\n\n".join(all_sheets_text)
