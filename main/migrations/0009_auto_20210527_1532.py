# Generated by Django 3.2.1 on 2021-05-27 15:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0008_auto_20210527_0650'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Attribute',
        ),
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
    ]
