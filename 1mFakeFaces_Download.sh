#!/bin/bash

download_link="https://archive.org/download/1mFakeFaces/1m_faces_00.tar"

destination_folder="./Data/downloaded_images"

mkdir -p "$destination_folder"

wget "$download_link" -P "$destination_folder"

if [ $? -eq 0 ]; then
    echo "Download successful!"
else
    echo "Download failed."
fi