#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup of a GCP project for the GPU CI. Run it as a project owner.
#
# usage: setup_project.sh <project-id>
#
# Creates:
#   - service account kvcached-ci, allowed to create, delete and SSH into
#     (through IAP, with sudo for Docker) only VMs and disks whose names start
#     with kvcached-ci-, plus the read-only and network permissions that
#     creating such a VM needs, which IAM conditions cannot scope by name;
#   - a Workload Identity pool that lets only workflows of
#     ovg-project/kvcached on refs/heads/main act as that service account, so
#     the workflows need no stored key;
#   - a firewall rule that admits SSH from the IAP range to VMs tagged
#     kvcached-ci.
# It prints the repository secrets and variables the workflows read.

set -euo pipefail

project=$1
repo=ovg-project/kvcached
sa_name=kvcached-ci
pool=github
provider=kvcached
sa=$sa_name@$project.iam.gserviceaccount.com
project_number=$(gcloud projects describe "$project" --format 'value(projectNumber)')

gcloud services enable compute.googleapis.com iap.googleapis.com oslogin.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com --project "$project"

gcloud iam service-accounts create "$sa_name" --project "$project" \
  --display-name "kvcached GPU CI" || true
# Only resources named kvcached-ci-*, so the CI cannot touch any other VM.
only_ci='resource.name.extract("/instances/{name}").startsWith("kvcached-ci-")'
only_ci+=' || resource.name.extract("/disks/{name}").startsWith("kvcached-ci-")'
for role in roles/compute.instanceAdmin.v1 roles/iap.tunnelResourceAccessor \
    roles/compute.osAdminLogin; do
  gcloud projects add-iam-policy-binding "$project" --quiet \
    --member "serviceAccount:$sa" --role "$role" \
    --condition "expression=$only_ci,title=kvcached-ci-only" >/dev/null
done
# What creating and polling such a VM needs besides that: using the subnet,
# reading zones, machine types and operations, listing instances.
gcloud iam roles create kvcachedCiSupport --project "$project" \
  --title "kvcached CI support" --stage GA \
  --permissions compute.subnetworks.use,compute.subnetworks.useExternalIp,\
compute.zoneOperations.get,compute.instances.list,compute.projects.get,compute.zones.get,\
compute.zones.list,compute.machineTypes.get,compute.regions.get || true
gcloud projects add-iam-policy-binding "$project" --quiet --condition=None \
  --member "serviceAccount:$sa" --role "projects/$project/roles/kvcachedCiSupport" >/dev/null

gcloud iam workload-identity-pools create "$pool" --project "$project" \
  --location global --display-name "GitHub Actions" || true
gcloud iam workload-identity-pools providers create-oidc "$provider" --project "$project" \
  --location global --workload-identity-pool "$pool" \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition "assertion.repository == '$repo' && assertion.ref == 'refs/heads/main'" || true
gcloud iam service-accounts add-iam-policy-binding "$sa" --project "$project" --quiet \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$project_number/locations/global/workloadIdentityPools/$pool/attribute.repository/$repo" \
  >/dev/null

gcloud compute firewall-rules create kvcached-ci-iap-ssh --project "$project" \
  --network default --direction INGRESS --allow tcp:22 \
  --source-ranges 35.235.240.0/20 --target-tags kvcached-ci || true

cat <<EOF

Set these repository secrets (Settings > Secrets and variables > Actions > Secrets),
so that the public logs do not show them:
  GCP_PROJECT              $project
  GCP_WIF_PROVIDER         projects/$project_number/locations/global/workloadIdentityPools/$pool/providers/$provider
  GCP_CI_SERVICE_ACCOUNT   $sa
and these repository variables (... > Variables):
  GCP_L4_ZONES             comma-separated zones with L4 quota, e.g. us-central1-a,us-central1-b
  GCE_IMAGE_FAMILY         an image family with NVIDIA driver >= 580, Docker optional
  GCE_IMAGE_PROJECT        the project that publishes that family
EOF
