import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image as PILImage
import base64

# Function to process and unpivot the data
def process_file(uploaded_file):
    # Load the uploaded Excel file
    dfs = pd.read_excel(uploaded_file, header=[0, 1], sheet_name=None)
    
    # Process each sheet separately
    processed_dfs = []
    for sheet_name, df in dfs.items():
        # Sort the MultiIndex to avoid UnsortedIndexError
        df = df.sort_index(axis=1)

        # Resetting the index to handle MultiIndex columns correctly
        df.columns = df.columns.droplevel(0)

        # Slicing the DataFrame to select specific rows (June 2023 to March 2024)
        df_selected = df.loc[2:13, :]

        # Selecting the last 8 columns (adjust as needed)
        df_selected = df_selected.iloc[:, -8:]

        # Adding a column for the sheet name
        df_selected['Sheet Name'] = sheet_name

        # Append the processed DataFrame to the list
        processed_dfs.append(df_selected)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df = combined_df.dropna(thresh=5)

    # Debug: Print the columns of combined_df
    print("Combined DataFrame columns:", combined_df.columns)

    # Assuming the 'Month' is already a column and unpivot the remaining columns
    # Adjust the column name if necessary
    columns_to_unpivot = [col for col in combined_df.columns if col != 'Month']  # Adjust if 'Month' has a different name

    if 'Month' not in combined_df.columns:
        st.warning("'Month' column not found. Please check the input data.")
        return None

    unpivoted_df = pd.melt(
        combined_df,
        id_vars=['Sheet Name', 'Month'],  # Keep 'Sheet Name' and 'Month' as identifier columns
        value_vars=columns_to_unpivot,    # Columns to unpivot
        var_name='Waste Type',            # New column for the type of waste or data type
        value_name='Waste Amount'         # New column for the values
    )

    # Rename columns and format the DataFrame
    column_mapping = {
        'Waste Type': 'Source Sub Type',
        'Sheet Name': 'Facility',
        'Month': 'Res_Date',
        'Waste Amount': 'Activity'
    }

    unpivoted_df = unpivoted_df.dropna(subset="Waste Amount")
    unpivoted_df.rename(columns=column_mapping, inplace=True)
    unpivoted_df['CF Standard'] = "IPCCC"
    unpivoted_df['Activity Unit'] = "m3"
    unpivoted_df['Gas'] = "CO2"

    # Reorder columns
    new_order = ['Res_Date', 'Facility', 'Source Sub Type', 'Activity', 'Activity Unit', 'CF Standard', 'Gas']
    unpivoted_df = unpivoted_df[new_order]

    return unpivoted_df

# Function to convert DataFrame to Excel and return as a downloadable file
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# Function to resize and return an image
def resize_image(image_path, width):
    img = PILImage.open(image_path)
    aspect_ratio = img.height / img.width
    new_height = int(width * aspect_ratio)
    img_resized = img.resize((width, new_height))
    
    # Save the resized image to a BytesIO object
    buffered = BytesIO()
    img_resized.save(buffered, format="PNG")
    
    # Encode the image to Base64
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

# Streamlit app
st.markdown(
    """
    <style>
    .centered-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 300px; /* Adjust width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add an image at the top of the app with reduced size
top_image_base64 = resize_image("image (3).png", width=300)  # Adjust width as needed
st.markdown(
    f'<img src="data:image/png;base64,{top_image_base64}" class="centered-image">',
    unsafe_allow_html=True
)

st.title("Vessel Waste Data Processor")
# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    # Process the uploaded file
    unpivoted_df = process_file(uploaded_file)
    
    # Display success message and show the processed DataFrame
    st.success("File processed successfully!")
    st.dataframe(unpivoted_df)

    # Button to download the processed file
    processed_data = to_excel(unpivoted_df)
    st.download_button(
        label="Download Processed Excel",
        data=processed_data,
        file_name='Processed_Oil_Generated.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
