#!/bin/bash

destination_folder="./Data/downloaded_images/1mFakeFaces"
mkdir -p "$destination_folder"

for i in {00..06}; do
    download_link="https://archive.org/download/1mFakeFaces/1m_faces_$i.tar"
    wget "$download_link" -P "$destination_folder"

    if [ $? -eq 0 ]; then
        echo "Download of 1m_faces_$i.tar successful!"
    else
        echo "Download of 1m_faces_$i.tar failed."
    fi
done
