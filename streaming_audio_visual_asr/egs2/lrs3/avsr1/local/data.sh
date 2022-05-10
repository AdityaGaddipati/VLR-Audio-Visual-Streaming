#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

. ./db.sh
. ./path.sh
. ./cmd.sh

download_and_untar=false
mp4_to_wav=true

# Manually fill the lrs3_username, lrs3_password
lrs3_username=
lrs3_password=

# Multimodal
audio_visual=false
mouth_roi=false

# Read Arguments
while true; do
    [ -z "${1:-}" ] && break;  # break if there are no arguments
    case "$1" in
        --*)
        name=`echo "$1" | sed s/^--// | sed s/-/_/g`;
        eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;

        oldval="`eval echo \\$$name`";
        # Work out whether we seem to be expecting a Boolean argument.
        if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then
        was_bool=true;
        else
        was_bool=false;
        fi
        eval $name=\"$2\";
        if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
        fi
        shift 2;
        ;;
        *) break;
    esac
done

if [ $# -gt  1]; then
    log "Error: No positional arguments are required."
    exit 1
fi

if [ -z "${LRS3}" ]; then
    log "Fill the value of 'LRS3' of db.sh"
    exit 1
fi

if $download_and_untar; then
    log "Downloading and Untarring the LRS3 with username ${lrs3_username} and passwoed ${lrs3_password}."
    local/download_and_untar.sh --remove-archive ${LRS3} ${lrs3_username} ${lrs3_password}
fi

# if $mp4_to_wav; then
#     log "Extracting .wav files from .mp4 files and storing it under the same directory"
#     local/mp4_to_wav.sh ${LRS3}
# fi

# generate mouth roi cropped videos
if ${mouth_roi}; then
    log "Extracting mouth_roi.mp4 files and storing it under the same directory"
    if ! [ -f ${face_predictor_path} ]; then
        wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O "${face_predictor_path}.bz2"
        bzip2 -d "${face_predictor_path}.bz2"
    fi
    if ! [ -f ${mean_face_path} ]; then
        wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O ${mean_face_path}
    fi
    python3 ./local/mouth_roi.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test --face_predictor_path ${face_predictor_path} --mean_face_path ${mean_face_path} --ffmpeg_path ${ffmpeg_path}
fi


# Make the Folders where ESPNet data-prep files will be stored
for dataset in train dev test; do
    log "Creating the ./data/${dataset} folders"
    mkdir -p ./data/${dataset}
done

# generate the utt2spk, wav.scp and text files
log "Generating the utt2spk, wav.scp and text files"
python3 ./local/data_prep.py --train_val_path ${LRS3}/trainval --test_path ${LRS3}/test --audio_visual ${audio_visual}
 
log "Generating the spk2utt files"
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

log "Fix sorting issues by calling fix_data_dir.sh"
utils/fix_data_dir.sh data/train
utils/fix_data_dir.sh data/test
utils/fix_data_dir.sh data/dev

log "Validate the data directory"
utils/validate_data_dir.sh data/train --no-feats
utils/validate_data_dir.sh data/test --no-feats
utils/validate_data_dir.sh data/dev --no-feats

