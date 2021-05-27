# Generated by Django 3.2.1 on 2021-05-27 15:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0009_auto_20210527_1532'),
    ]

    operations = [
        migrations.AlterField(
            model_name='log',
            name='title',
            field=models.CharField(max_length=255, verbose_name='title'),
        ),
        migrations.AlterField(
            model_name='log',
            name='user_id',
            field=models.CharField(max_length=255, verbose_name='user_id'),
        ),
        migrations.AlterField(
            model_name='user',
            name='id',
            field=models.CharField(max_length=255, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='user',
            name='login',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='user',
            name='password',
            field=models.CharField(max_length=255),
        ),
    ]