import mimetypes

from django.contrib.auth.decorators import login_required
from django.http import Http404, response, HttpResponse
from django.shortcuts import render, redirect

from .diploma_pipline import run_pipline, visualize_petri_net_color_clusters, save_visualization, pnml_exporter
from .models import Log
from .forms import LogForm, UserForm
from django.views import generic
from .handle_log_file import parse_log
from django.db.models import Q
import base64


class LogDetailView(generic.DetailView):
    model = Log

@login_required
def log_detail_view(request, pk):
    try:
        log_id = Log.objects.get(pk=pk)
    except Log.DoesNotExist:
        raise Http404("Log does not exist")

    return render(
        request,
        'main/log_detail.html',
        context={'log': log_id, }
    )

@login_required
def index(request):
    # for obj in Log.objects.filter(user_id=request.user).all():
    #     if obj.n_traces is None:
    #         obj.n_traces = 5
    # logs = Log.objects.filter(user_id=request.user.username).order_by('title')
    logs = Log.objects.filter(Q(user_id=request.user.username) | Q(user_id="admin")).order_by("title")
    context = {
        'logs': logs
    }
    return render(request, 'main/index.html', context)

def about(request):
    return render(request, 'main/about.html')

@login_required
def add_log(request):
    error = ''
    if request.method == 'POST':
        form = LogForm(request.POST, request.FILES)
        try:
            log, n_traces, attributes = parse_log(request.FILES["xes_file"])
            new_vals = dict()
            new_vals["attributes"] = attributes
            new_vals["n_traces"] = n_traces
            new_vals["csrfmiddlewaretoken"] = request.POST["csrfmiddlewaretoken"]
            new_vals["title"] = request.POST["title"]
            new_vals["user_id"] = request.user.username
            form = LogForm(new_vals, request.FILES)

            if form.is_valid():
                form.save()
                return redirect('home')
            else:
                error = "Ошибка при заполнении формы!"
        except (NameError, OSError) as err:
            if type(err) == NameError:
                raise Http404(err.args[0])
            else:
                raise OSError

    else:
        form = LogForm()

    context = {
        'form': form,
        'error': error
    }
    return render(request, 'main/add_log.html', context)

@login_required
def delete_log(request, pk):
    object = Log.objects.get(pk=pk)
    object.delete()
    return redirect('home')

@login_required
def run(request, pk):
    object = Log.objects.get(pk=pk)
    title = object.title
    xes_file = object.xes_file

    net, im, fm, activity_to_cluster = run_pipline(xes_file.name)

    gviz = visualize_petri_net_color_clusters(net, im, fm, activity_to_cluster)
    save_visualization(gviz, f"{pk}.png")
    pnml_exporter.apply(net, im, f"{pk}.pnml", final_marking=fm)
    with open(f"{pk}.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    return render(
        request,
        'main/activate_details.html',
        context={'image': image_data,
                 'title': title,
                 'pk': pk}
    )

@login_required
def download(request, pk):
    # object = Log.objects.get(pk=pk)
    #
    # title = object.title
    # xes_file = object.xes_file
    #
    # net, im, fm, activity_to_cluster = run_pipline(xes_file.name)
    # pnml_exporter.apply(net, im, f"{pk}.pnml", final_marking=fm)
    fsock = open(f"{pk}.pnml", "rb")
    response = HttpResponse(fsock)
    response['Content-Disposition'] = f'attachment; filename={pk}.pnml'
    return response


def auth(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            try:
                form.save()
                return redirect('main/index.html')
            except:
                pass
    else:
        form = UserForm()
    return render(request, 'main/auth.html', {'form': form})
