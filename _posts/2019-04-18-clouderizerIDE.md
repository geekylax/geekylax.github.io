---
layout: post
title:  "Using  Cluderizer IDE "
date:   2019-04-15 18:34:04 +0530
categories: jekyll update
tags: clouderizer,deeplearniig,colab
---
The Fastest way to train test and deploy machine learning and deep learning models

While doing my tutorials and research in machine learning and deep learning i just search for free cloud service that i can utilize to Train Test and Deploy my model

When free cloud providers for certain month came, i just go through it and test it with my models

[![Dean commits all]({{ site.url }}/img/secondblog/first.jpeg)]({{ site.url }}/img/secondblog/first.jpeg)

<!-- 
[!free]({{ site.url }}/img/secondblog/first.jpeg)]({{ site.url }}/img/secondblog/first.jpeg)
 -->

**Where every cloud providers giving free credits to use their service if its fine go for it otherwise remove the account**


[![cloud]({{ site.url }}/img/secondblog/second.jpeg)]({{ site.url }}/img/secondblog/second.jpeg)

But when comes to research in Machine Learning or Deep learning the way of learning training and testing your model should not be much more complicated


__COLAB__

Here the google colab came for using the GPU and as well as TPU engine for free with some bugs 🤣

[![colab]({{ site.url }}/img/secondblog/third.jpeg)]({{ site.url }}/img/secondblog/third.jpeg)


But it comes with some bugs and we its quite complicated using colab to write code to mount drive blaa…bla….bla..

And AWS Azure IBM we used it for but colab is free and as well as Kaggle Kernels

__Clouderizer__

Here comes Clouderizer i just came through a blog and notifies clouderizer IDE to use Colab AWS Kaggle Kernels and own ubuntu machine


[![clouderizer]({{ site.url }}/img/secondblog/fourth.jpeg)]({{ site.url }}/img/secondblog/fourth.jpeg)


They use tensorboard gui and google drive integrated

we can use Clouderizer for free for first month and afterward 4.99 dollar based on the projects used it varies

So i just tried to use fast.ai its already prebuilt with a template

This is the first time i just used to setup fastai within 1 min using colab

**COLAB,fastai + Clouderizer**


[![colab,fastai]({{ site.url }}/img/secondblog/fifth.jpeg)]({{ site.url }}/img/secondblog/fifth.jpeg)


At first once we login to our account we should ensure authentication with google drive to save the files


Then we have to setup a resource we want to work on it here i am taking fastai template

They have fast ai template in public Templates folder


[![templates]({{ site.url }}/img/secondblog/sixth.jpeg)]({{ site.url }}/img/secondblog/sixth.jpeg)


There will be a lots of templates available here you can see fastai template here

[![fastai]({{ site.url }}/img/secondblog/seventh.jpeg)]({{ site.url }}/img/secondblog/seventh.jpeg)


Clone the template and use that


[![clone]({{ site.url }}/img/secondblog/eight.jpeg)]({{ site.url }}/img/secondblog/eight.jpeg)


Start the machine and it will ask which resource that we have to use


We can use any resouces we want and also they have their own resources form Clouderizer too as ubuntu machine

Here i am using colab

It will redirect to google colab get authenticate and come to clouderizer back


ssh and jupyter notebook

Here you can see in top left corner both ssh and jupter notebook

If we want to do changes in ssh or download files we can use ssh also

In Juypter notebook i am running pest classification notebook from fastai

[![jupyter]({{ site.url }}/img/secondblog/ninth.jpeg)]({{ site.url }}/img/secondblog/ninth.jpeg)


And the ui of jupyterlab will be like this

__ DEPLOY MODEL __

To deploy a model we have to write an rest api and want to seve with a web application

Here we can do that with a single click

While setup the machine we can see this tab


[![setup]({{ site.url }}/img/secondblog/tenth.jpeg)]({{ site.url }}/img/secondblog/tenth.jpeg)


Here we can see cloud serving check box click that

They have prebuilt object detection tensorflow template in public templates clone that and start cloud serving

here im serving with colab itself if you have AWS acoount you can setup with AWS too


[![colab]({{ site.url }}/img/secondblog/eleventh.jpeg)]({{ site.url }}/img/secondblog/eleventh.jpeg)


Ignore my face “sorry”
The api predicted succesfully that i am a person “hurray”

here i have used webcam you can predict it with your image from local file

And we can deploy our own api with clud serving here

Here you can see the methods in the referenced articles


__ References __


Fastest way to serve, demo and test your Deep Learning Model with a mobile web app and REST APIs [here](https://towardsdatascience.com/fastest-way-to-serve-demo-and-test-your-deep-learning-model-with-a-mobile-web-app-and-rest-apis-38cfbbf90a52)


Fastest way to setup Fast.ai v3 2019 course notebooks — using Google Colab and Clouderizer [here](https://medium.com/@prakash_31206/fastest-way-to-setup-fast-ai-course-notebooks-for-free-using-google-colab-gpu-and-clouderizer-c8a004e1d50d)




I am currently working with Mask Rcnn with 3d images you can see my projects and upcoming researches in my github repos

https://github.com/geekylax


