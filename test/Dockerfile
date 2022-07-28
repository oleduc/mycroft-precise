# This dockerfile is for continuous integration of the mycroft-precise repostiory

# Build an environment that can run the Precise wake word spotter.
FROM python:3.9-slim as precise-build
ENV TERM linux
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    build-essential git python3-scipy cython3 libhdf5-dev python3-h5py \
    portaudio19-dev swig libpulse-dev libatlas-base-dev \
    alsa-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /root/allure /opt/mycroft/mycroft-precise /root/code-quality
WORKDIR /opt/mycroft
COPY test/requirements.txt mycroft-precise/test/
RUN python3 -m pip install --no-cache-dir -r mycroft-precise/test/requirements.txt
COPY . mycroft-precise
RUN python3 -m pip install --no-cache-dir -e ./mycroft-precise/
# Run the tests defined in the precise repository
FROM precise-build as test-runner
WORKDIR /opt/mycroft/mycroft-precise
RUN python3 -m pip install --no-cache-dir ./runner
ENTRYPOINT ["python3", "-m", "pytest"]

FROM precise-build as precise-codequality
RUN python3 -m pip install --no-cache-dir pylint2codeclimate pylint
ENTRYPOINT ["python3", "-m", "pylint"]

# Do it at the end to prevent all these to be executed when we cannot access devops repository
#
# Clone the devops repository, which contiains helper scripts for some continuous
# integraion tasks. Run the code_check.py script which performs linting (using PyLint)
# and code formatting (using Black)
FROM precise-build as code-checker
ARG github_api_key
ENV GITHUB_API_KEY=$github_api_key
RUN pip install --no-cache-dir pipenv
RUN git clone https://$github_api_key@github.com/MycroftAI/devops.git
WORKDIR /opt/mycroft/devops/jenkins
RUN pipenv install
ENTRYPOINT ["pipenv", "run", "python","-m", "pipeline.code_check", "--repository", "mycroft-precise", "--pull-request", "PR-149"]
