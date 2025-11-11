

def addCategory(columns, name, data, axis):
    col = Column(fields=("Categories", name))
    columns[col] = data
    to_add = CategoryDesc(column=col, axis=axis)
    if "categories" not in columns.pipeline_data:
        columns.pipeline_data["categories"] = []
    columns.pipeline_data["categories"].append(to_add)
    return columns
