
#!bin/bash
#----------------------------------------------------#
# Download data from kaggle Dataset,
# one has to configure the API and save the kaggle.json file
# under home directory to be able to use this script.
#----------------------------------------------------#
DATA_NAME="imdb-dataset.csv"
DATA_ZIP="imbd-movie-reviews-for-binary-sentiment-analysis.zip"
DATA_DIR="data"

if test -e ${DATA_ZIP}; then
    rm ${DATA_ZIP}
fi

kaggle datasets download -d mwallerphunware/imbd-movie-reviews-for-binary-sentiment-analysis


unzip -p ${DATA_ZIP} > ${DATA_NAME}
rm ${DATA_ZIP}

if [ -d ${DATA_DIR} ]; then
    echo "${DATA_DIR} already exists."
else
    mkdir ${DATA_DIR}
fi

echo "Copying ${DATA_NAME} into ${DATA_DIR} folder."
mv ${DATA_NAME} ${DATA_DIR}
