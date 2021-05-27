from django.db import models
from django.urls import reverse


class User(models.Model):
    id = models.CharField(max_length=20, primary_key=True)
    login = models.CharField(max_length=100)
    password = models.CharField(max_length=100)

    class Meta:
        db_table = "users"

#
# class Attribute(models.Model):
#     """
#     Model representing a book genre (e.g. Science Fiction, Non Fiction).
#     """
#     name = models.CharField(max_length=20)
#
#     def __str__(self):
#         """
#         String for representing the Model object (in Admin site etc.)
#         """
#         return self.name
#

class Log(models.Model):
    user_id = models.CharField("user_id", max_length=100)
    title = models.CharField("title", max_length=100)
    xes_file = models.FileField("xes_file", upload_to="logs", help_text="Выберите файл")
    n_traces = models.IntegerField("n_traces", null=True)
    attributes = models.TextField("attributes", null=True)

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        """
        Returns the url to access a particular book instance.
        """
        return reverse('log-detail', args=[str(self.id)])

    def get_absolute_url_for_delete(self):
        return reverse('delete_view', args=[str(self.id)])

    def get_absolute_url_for_run(self):
        return reverse('log', args=[str(self.id)])

    class Meta:
        verbose_name = "Лог"
        verbose_name_plural = "Логи"


