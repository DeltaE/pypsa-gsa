# Developer Depolyment Instructions

The dashboard is hosted through Google Cloud Run. To update the dashboard, ensure the [Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk#linux) is installed. Detailed instructions can be found [here](https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service).

Authenticate with Google Cloud:

```bash 
gcloud auth login
```

Set the project ID: 

```bash 
gcloud config set project pypsa-usa-489820
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
docker build . --tag us-central1-docker.pkg.dev/pypsa-usa-489820/pypsa-usa/app:v[VERSION]
```

Push to the cloud: 

```bash 
docker push us-central1-docker.pkg.dev/pypsa-usa-489820/pypsa-usa/app:v[VERSION]
```

Deploy to Cloud Run. Cached data can be stored via a Redis instance. Do not use a Google Cloud Storage bucket as the caching exceeds the 1 file write per second and can lead to crashes. 

Set the `REDIS_URL` environment variable to your Redis connection string (ie. starts with `rediss://`). When `REDIS_URL` is set, the dashboard uses `RedisBackend` for server-side caching. Depoly to Google Cloud Run. 

```bash 
gcloud run deploy dash-service \
    --image us-central1-docker.pkg.dev/pypsa-usa-489820/pypsa-usa/app:v[VERSION] \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 8Gi \
    --session-affinity \
    --set-env-vars=REDIS_URL=redis://[HOST]:[PORT]
```

### Troubleshooting

If you get a `error from registry: Unauthenticated request.` error, try running the following command:

```bash 
sudo usermod -aG docker $USER # Add user to docker group
newgrp docker # Apply changes
```

And then reauthenticate with google cloud:

```bash 
gcloud auth configure-docker us-central1-docker.pkg.dev
```
