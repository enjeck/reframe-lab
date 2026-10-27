#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$ROOT/.aws-deploy.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.aws-deploy.env"
fi
STACK_NAME=${STACK_NAME:-reframe-lab}
AWS_REGION=${AWS_REGION:-us-east-1}
REPOSITORY_NAME=${REPOSITORY_NAME:-reframe-lab-worker}
BUCKET=${JOB_BUCKET:-$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" --query 'Stacks[0].Outputs[?OutputKey==`JobBucketName`].OutputValue' --output text 2>/dev/null || true)}
if [[ -n "$BUCKET" && "$BUCKET" != "None" ]]; then
  aws s3 rm "s3://$BUCKET" --recursive --region "$AWS_REGION" || true
fi
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
if [[ "${DELETE_ECR_REPOSITORY:-0}" == "1" ]]; then
  aws ecr delete-repository --repository-name "$REPOSITORY_NAME" --force --region "$AWS_REGION" || true
fi
rm -f "$ROOT/.aws-deploy.env"
echo "Deleted stack $STACK_NAME."
