# Generated by Django 3.2.3 on 2021-05-25 11:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0005_alter_log_log'),
    ]

    operations = [
        migrations.AlterField(
            model_name='log',
            name='log',
            field=models.TextField(null=True, verbose_name='log'),
        ),
    ]
