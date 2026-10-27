#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$ROOT/.aws-deploy.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.aws-deploy.env"
fi
STACK_NAME=${STACK_NAME:-reframe-lab}
AWS_REGION=${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}
AWS_REGION=${AWS_REGION:-us-east-1}
REPOSITORY_NAME=${REPOSITORY_NAME:-reframe-lab-worker}
IMAGE_TAG=${IMAGE_TAG:-0.14.0-$(date +%Y%m%d%H%M%S)}
ACCESS_CODE=${ACCESS_CODE:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')}
RETENTION_DAYS=${RETENTION_DAYS:-1}
RETENTION_HOURS=${RETENTION_HOURS:-12}
MAX_UPLOAD_BYTES=${MAX_UPLOAD_BYTES:-4294967296}
DOWNLOAD_CORE_MODELS=${DOWNLOAD_CORE_MODELS:-1}

for command in aws docker sam python3; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 2; }
done
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE_TAG_URI="$REGISTRY/$REPOSITORY_NAME:$IMAGE_TAG"

if ! aws ecr describe-repositories --repository-names "$REPOSITORY_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws ecr create-repository \
    --repository-name "$REPOSITORY_NAME" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --region "$AWS_REGION" >/dev/null
fi

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

cd "$ROOT"
rm -rf dist build .aws-sam
python3 -m pip wheel . --no-deps --wheel-dir dist

docker buildx build \
  --platform linux/amd64 \
  --build-arg DOWNLOAD_CORE_MODELS="$DOWNLOAD_CORE_MODELS" \
  --file deploy/aws/Dockerfile \
  --tag "$IMAGE_TAG_URI" \
  --load \
  .
docker push "$IMAGE_TAG_URI"
IMAGE_DIGEST=None
for attempt in {1..10}; do
  IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name "$REPOSITORY_NAME" \
    --image-ids imageTag="$IMAGE_TAG" \
    --region "$AWS_REGION" \
    --query 'imageDetails[0].imageDigest' \
    --output text 2>/dev/null || true)
  if [[ -n "$IMAGE_DIGEST" && "$IMAGE_DIGEST" != "None" ]]; then
    break
  fi
  sleep 2
done
if [[ -z "$IMAGE_DIGEST" || "$IMAGE_DIGEST" == "None" ]]; then
  echo "Could not resolve the pushed ECR image digest." >&2
  exit 2
fi
IMAGE_URI="$REGISTRY/$REPOSITORY_NAME@$IMAGE_DIGEST"

# The API function has no third-party dependencies (boto3 ships with the Lambda
# runtime), so there is no `sam build` step. That step would demand a local
# python3.12. `sam deploy` zips and uploads api/ directly.
sam deploy \
  --template-file deploy/aws/template.yaml \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    WorkerImageUri="$IMAGE_URI" \
    AccessCode="$ACCESS_CODE" \
    MaxUploadBytes="$MAX_UPLOAD_BYTES" \
    RetentionDays="$RETENTION_DAYS" \
    RetentionHours="$RETENTION_HOURS"

WEB_URL=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`WebUrl`].OutputValue' \
  --output text)
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].Outputs[?OutputKey==`JobBucketName`].OutputValue' \
  --output text)

# Pre-rendered sample for visitors who do not want to upload a video.
if [[ -d "$ROOT/deploy/aws/sample" ]]; then
  aws s3 sync "$ROOT/deploy/aws/sample/" "s3://$BUCKET/samples/" \
    --region "$AWS_REGION" --only-show-errors
fi

{
  printf 'STACK_NAME=%q\n' "$STACK_NAME"
  printf 'AWS_REGION=%q\n' "$AWS_REGION"
  printf 'REPOSITORY_NAME=%q\n' "$REPOSITORY_NAME"
  printf 'IMAGE_TAG_URI=%q\n' "$IMAGE_TAG_URI"
  printf 'IMAGE_URI=%q\n' "$IMAGE_URI"
  printf 'WEB_URL=%q\n' "$WEB_URL"
  printf 'JOB_BUCKET=%q\n' "$BUCKET"
  printf 'ACCESS_CODE=%q\n' "$ACCESS_CODE"
} > "$ROOT/.aws-deploy.env"
chmod 600 "$ROOT/.aws-deploy.env"

printf '\nDeployment complete.\n\nWeb URL     %s\nAccess code %s\nJob bucket  %s\n\nSaved to %s/.aws-deploy.env\n' \
  "$WEB_URL" "$ACCESS_CODE" "$BUCKET" "$ROOT"
