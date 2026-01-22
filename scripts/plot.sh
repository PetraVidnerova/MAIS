CONFIG=../config/higgs_sir.ini
EXP_ID=higgs_sir20
COLUMN=inc_I

FILENAMES=""
LABELS=""
for FILE in `ls ../data/output/model/history_${EXP_ID}_*.zip`
do
    FILENAMES="$FILENAMES $FILE"
    LABEL=`basename $FILE .zip`
    LABELS="${LABELS},$LABEL"
done
LABELS=`echo $LABELS | sed 's/^,//g' | eval sed 's/history_${EXP_ID}_//g'`

#python plot_experiments.py $FILENAMES --label_names $LABELS --out_file ${EXP_ID}.png --column ${COLUMN} --fit_me ../data/fit_data/retweets.csv --ymax 10000
python plot_experiments.py $FILENAMES --out_file ${EXP_ID}.png --column ${COLUMN} --fit_me ../data/fit_data/retweets.csv --ymax 40000
#geeqie ${EXP_ID}.png
