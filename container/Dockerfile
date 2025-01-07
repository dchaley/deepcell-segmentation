# Base container image is the *rebuilt* tensorflow 2.8.4-gpu image.
# See also: https://github.com/dchaley/tensorflow-2.8.4-redux
ARG GCP_LOCATION=us-central1
ARG GCP_PROJECT_ID=my-project
ARG GCP_REPOSITORY=my-repository
ARG BASE_IMAGE=${GCP_LOCATION}-docker.pkg.dev/${GCP_PROJECT_ID}/${GCP_REPOSITORY}/tensorflow:2.17.0-gpu
FROM ${BASE_IMAGE}

# Update installed packages & install graphviz & git
# Clear out the aptitude cache as well.
# Lastly, install and/or update pip.
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends git google-cloud-sdk google-cloud-cli pigz python3.10 python3-pip python3-dev python3.10-distutils && \
    apt-get remove -y python3.11 && \
    rm -rf /var/lib/apt/lists/* && \
    python3.10 -m pip install --no-cache-dir --upgrade pip && \
    gcloud config set storage/parallel_composite_upload_enabled True

# Pull the requirements.txt file.
# FIXME: we should be using our local copy of the requirements file & source.
# The way it is now, we can only build from the main branch.
# Changing this means rearranging how we build, so skipping for now.
ADD requirements.txt .
ADD requirements-no-deps.txt .

# Install python requirements
# This one is for packages we need, but which specify outdated dependencies.
RUN python3.10 -m pip install --user --upgrade --no-deps -r requirements-no-deps.txt --no-warn-script-location
# This one we install as usual.
RUN python3.10 -m pip install --user --upgrade -r requirements.txt --no-warn-script-location

WORKDIR "/deepcell-segmentation"
COPY src src
COPY scripts scripts
COPY pyproject.toml .
COPY requirements.txt .
COPY setup.py .

# Install our own module
RUN pip install .

# The container entrypoint is the benchmark script.
# Command-line arguments go to the script.
ENTRYPOINT ["python", "benchmarking/deepcell-e2e/benchmark.py"]
