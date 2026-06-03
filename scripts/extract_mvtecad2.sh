#!/bin/bash
#SBATCH --job-name=dl-mvtecad2
#SBATCH --output=/shared/home/juan.osorio/ml/logs/dl-mvtecad2-%j.out
#SBATCH --error=/shared/home/juan.osorio/ml/logs/dl-mvtecad2-%j.err
#SBATCH --partition=amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# Download MVTec AD 2 per-category (full tarball is corrupt on mydrive.ch)
# Usage: sbatch extract_mvtecad2.sh

DEST="/shared/home/juan.osorio/ml/data/mvtec_ad_2"

echo ">>> Cleaning old partial extraction..."
chmod -R u+w "${DEST}" 2>/dev/null || true
rm -rf "${DEST}"
mkdir -p "${DEST}"

echo ">>> Disk space:"
df -h /shared/home/juan.osorio/ml/data

declare -a CATEGORIES=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)
declare -a URLS=(
    "https://www.mydrive.ch/shares/121501/26456e2f3ef813930866f8f9b072593a/download/466651130-1743159807/can.tar.gz"
    "https://www.mydrive.ch/shares/150467/1a5fe9cc26e9225886bef53d0b645bb8/download/466651519-1743162446/fabric.tar.gz"
    "https://www.mydrive.ch/shares/121503/951a46ce30a3af3787ce9671cfa8613a/download/466651800-1743164023/fruit_jelly.tar.gz"
    "https://www.mydrive.ch/shares/121504/0014676292c3c44931712a54fb3bdbe8/download/466653907-1743164943/rice.tar.gz"
    "https://www.mydrive.ch/shares/121505/2d8fcdc8e988456bdd18696746eda0a0/download/466654829-1743166795/sheet_metal.tar.gz"
    "https://www.mydrive.ch/shares/121506/739dc6459c939fe464c0d26acc6c2d55/download/466654885-1743167505/vial.tar.gz"
    "https://www.mydrive.ch/shares/121507/66fe6e114b498e03be8d48c711794be7/download/466655287-1743168151/wallplugs.tar.gz"
    "https://www.mydrive.ch/shares/121508/9fcf67e49f0dc61a9608f57ba0482356/download/466656233-1743168988/walnuts.tar.gz"
)

for i in "${!CATEGORIES[@]}"; do
    CAT="${CATEGORIES[$i]}"
    URL="${URLS[$i]}"

    if [ -d "${DEST}/${CAT}" ]; then
        echo ">>> ${CAT} already exists, skipping."
        continue
    fi

    echo ""
    echo "============================================"
    echo ">>> Downloading ${CAT}..."
    echo "============================================"
    if ! wget --no-check-certificate "${URL}" -O "${DEST}/${CAT}.tar.gz"; then
        echo ">>> FAILED to download ${CAT} (404 or network error). Skipping."
        rm -f "${DEST}/${CAT}.tar.gz"
        continue
    fi

    echo ">>> Extracting ${CAT}..."
    tar xzf "${DEST}/${CAT}.tar.gz" --no-same-permissions -C "${DEST}"
    rm "${DEST}/${CAT}.tar.gz"

    echo ">>> ${CAT} done."
    ls "${DEST}/${CAT}/" 2>/dev/null || echo "(extracted to different name?)"
done

echo ""
echo "============================================"
echo ">>> All categories:"
ls "${DEST}/"
du -sh "${DEST}/"
echo ">>> Complete."
