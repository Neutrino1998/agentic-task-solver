
def filter_workspace_content(workspace: dict, content_type):
    filtered_workspace = {
            key:value for key, value in workspace.items() if isinstance(value.get("content"), content_type)
        }
    return filtered_workspace

if __name__ == "__main__":
    import pandas as pd
    # =======================================================
    # Test Example
    sample_dataframe = pd.DataFrame({"region": ["North", "South"], "sales": [300, 400]})
    sample_text = "# My Article\nThis is a sample text."
    test_workspace = {
                        "sales_data": {
                            "content": sample_dataframe,
                            "metadata": {}
                            },
                        "my_article": {
                           "content": sample_text,
                           "metadata": {}
                       }
                      }
    filtered_workspace = filter_workspace_content(test_workspace, content_type=pd.DataFrame)
    print(filtered_workspace)