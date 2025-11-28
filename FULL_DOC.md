# About the team

The dream team is integrated by:

- Amir Sadour
- Camilo Rodriguez
- Claudia Agudelo
- Luis Manrique

# Repository

Your code could be found on [github](https://github.com/amir1226/animaldet)

# Live demo

You can see a demo in this [link](http://animaldet-alb-510958915.us-east-1.elb.amazonaws.com/)

# General animaldet installation guide

Animaldet was built as a containerized application which can run locally
or at any cloud which support containers

# Local deployment

First of all with the repo cloned you must execute 
```bash
git lfs pull
```

To get models and other useful information to deploy your model.

You should have `docker` installed.

After that you can execute

```bash
make build
docker run -p 8000:8000 animaldet:latest
```

Then when opening in your browser `localhost:8000` you must see the application running. 
![Main view](static/img/main_view.png)

For user guide about how to use Animaldet you can check the [next link](USER_GUIDE.md)
# Cloud deployment

Animaldet included deployment platform is `AWS ECS`, with the next architecture:

![Architecture](static/img/architecture.png)


We included all the required infrastructure as a code at [infra/aws](infra/aws), there we use `terraform`.

In order to deploy the application you must configure your aws credentials, more information at [aws cli configuration documentation](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html)

Commands to deploy the application 

```bash
make deploy
```

It will execute `docker build`, then `terraform apply` which is going to:

1. Build docker image locally
2. Login you to ECS registry
3. Push the image to ECS
4. Create clusters, task definitions, policies and load balancer to get the application working.

# Animaldet: how to use our animal detection tool

First of all, the main screen:

![main screen](static/img/main_view.png)

You may chose your own images or select one of the sample ones.

If you decide to pick one of the samples, you are going to 
see the image annotations and a button to predict executions:

![sample chosen](static/img/guide/sample_chosen.png)

You can also pick between `small` and `nano` model, and change the model confidence.

## Model size

Nano model uses 30% less ram than Small model, but as it uses
a lower resolution it must perform more inferences over the whole image, inference times does not differ much between those models.

## Confidence

Each model has its own suggested confidence, confidence is a model output where it told us how confident the model is about a found bounding box with its respective class, your recommended confidence is the one which maximizes performance but you may experiment with different values

## Performing inference

Once the image and parameters are chosen, you can click then `Detect Animals` button, and a loading screen must start

![alt text](static/img/guide/loading.png)

Once finished it must show the image detections:

![detections](static/img/guide/detections.png)


# Low confidence: over detection

When you pick a lower confidence you may expect over prediction as shown in the next capture:

![alt text](static/img/guide/overdetection.png)

It may work for experimentation purposes, but please be careful. 

# Experiment comparison: lets see how the model performed

We have added an experiments view where you can see detections
over the whole test dataset compared with the ground truth:


![experiments](static/img/guide/experiments.png)

There you can pick any image from the test set and check the difference between the ground truth and your predictions.