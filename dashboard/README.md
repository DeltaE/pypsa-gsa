# Developer Depolyment Instructions

The dashboard is hosted through Google Cloud Run. To update the dashboard, ensure the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk#linux) is installed. Detailed instructions can be found [here](https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service).

Authenticate with Google Cloud:

```bash 
gcloud auth login
```

Set the project ID: 

```bash 
gcloud config set project [PROJECT_ID]
```

Configure docker authentication:

```bash 
gcloud auth configure-docker us-central1-docker.pkg.dev
```

Create a repository in artifact registry:

```bash
    gcloud artifacts repositories create pypsa-usa \
        --repository-format=docker \
        --location=us-central1 \
        --description="PyPSA-USA Dashboard" \
        --immutable-tags \
        --async
```

Build and tag the image:

```bash 
docker build . --tag us-central1-docker.pkg.dev/[PROJECT_ID]/pypsa-usa/app:v[VERSION]
```

Push to the cloud: 

```bash 
docker push us-central1-docker.pkg.dev/[PROJECT_ID]/pypsa-usa/app:v[VERSION]
```

Deploy to Cloud Run. Cached data is stored in a Google Cloud Storage bucket.

```bash 
gcloud run deploy dash-service \
    --image us-central1-docker.pkg.dev/[PROJECT_ID]/pypsa-usa/app:v[VERSION] \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 8Gi \
    --session-affinity \
    --execution-environment gen2 \
    --add-volume=name=gcs_fuse_cache,type=cloud-storage,bucket=[BUCKET_NAME] \
    --add-volume-mount=volume=gcs_fuse_cache,mount-path=/mnt/dash_cache \
    --set-env-vars=CACHE_DIR=/mnt/dash_cache
```

If you get a `error from registry: Unauthenticated request.` error, try running the following command:

```bash 
sudo usermod -aG docker $USER # Add user to docker group
newgrp docker # Apply changes
```

And then reauthenticate with google cloud:

```bash 
gcloud auth configure-docker us-central1-docker.pkg.dev
```
