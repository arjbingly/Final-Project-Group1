#!/bin/bash

zip_url="https://storage.googleapis.com/project_deep_learning_team_1/Models.zip"

zip_file="Models.zip"

unzip_dir=“Models”

wget "$zip_url"

if [ $? -eq 0 ]; then
    echo "Download successful. Now unzipping..."

    unzip "$zip_file" -d "$unzip_dir"

    if [ $? -eq 0 ]; then
        echo "Unzip successful. Files are in the '$unzip_dir' directory."
    else
        echo "Error: Failed to unzip the file."
    fi
else
    echo "Error: Failed to download the zip file."
fi
