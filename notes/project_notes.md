------------ 4/18/2026 ------------
export NET_ID="chris-joe"
export BUCKET="${NET_ID}-datsbd-s2026-v2"
export DEST="s3://${BUCKET}/reddit-data/"

used the following copy command to copy the reddit data to our bucket:
aws s3 cp s3://adg-reddit-data/ ${DEST} --recursive --request-payer requester --copy-props none