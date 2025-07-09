import pandas as pd
import os

def load_datasets(data_path="../data/"):
    """
    Загружает и возвращает датасеты из Excel-файлов:
    - база знаний (Knowledge_base)
    - пары вопрос-ответ (Sources)

    Параметры
    ----------
    data_path : str
        Путь к директории с Excel-файлами.

    Возвращает
    ----------
    know_df : pd.DataFrame
        Датафрейм базы знаний.
    
    qa_pairs_df : pd.DataFrame
        Датафрейм с источниками (вопросы и ответы).
    """
    files = os.listdir(data_path)
    if len(files) < 2:
        raise ValueError("В папке должно быть как минимум два Excel-файла: база знаний и вопросы-ответы.")

    # Предполагаем, что первый файл — база знаний, второй — пары
    know_file_path = os.path.join(data_path, files[0])
    qa_pairs_file_path = os.path.join(data_path, files[1])

    know_xls = pd.ExcelFile(know_file_path)
    qa_pairs_xls = pd.ExcelFile(qa_pairs_file_path)

    # Превью каждого листа
    dfs = {}
    for sheet in know_xls.sheet_names:
        df = know_xls.parse(sheet)
        dfs[sheet] = df
    # Загружаем конкретные листы
    know_df = know_xls.parse("Knowledge_base")
    pairs_df = dfs['Sources']

    qa_pairs_xls.sheet_names
    # Распарсиваем и переводим в формат .csv
    qa_sheet = qa_pairs_xls.sheet_names[0]
    qa_pairs_df = qa_pairs_xls.parse(qa_sheet)

    # Заполняем пропуски в part_id медианой
    if 'part_id' in know_df.columns:
        know_df['part_id'].fillna(know_df['part_id'].median(), inplace=True)

    return know_df, pairs_df, qa_pairs_df
