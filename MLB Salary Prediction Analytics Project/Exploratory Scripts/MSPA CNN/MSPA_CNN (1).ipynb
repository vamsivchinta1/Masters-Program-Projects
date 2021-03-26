{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9P3BL4p6YXIK",
    "tags": []
   },
   "source": [
    "# Vamsi Chinta\n",
    "# Machine Learning - Facial Recognition Project\n",
    "# CNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 123
    },
    "collapsed": true,
    "id": "2S1uAQexAtfN",
    "jupyter": {
     "outputs_hidden": true
    },
    "outputId": "2ed15171-098e-4859-8a25-5e615a1d425c",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code\n",
      "\n",
      "Enter your authorization code:\n",
      "··········\n",
      "Mounted at /content/gdrive\n"
     ]
    }
   ],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/gdrive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "collapsed": true,
    "id": "qRX1FM-B0ny8",
    "jupyter": {
     "outputs_hidden": true
    },
    "outputId": "a909362e-b516-4df6-c3b8-975a443819af",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Once deleted, variables cannot be recovered. Proceed (y/[n])? y\n"
     ]
    }
   ],
   "source": [
    "%reset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "id": "klrcZaHsBu0n",
    "outputId": "8133bd73-d789-4db7-e325-67335d852852"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "import keras\n",
    "from keras import backend as K\n",
    "from keras import layers, models, Sequential\n",
    "from keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten\n",
    "from keras.layers import Conv2D as C2D\n",
    "from keras.layers import MaxPooling2D as MP2D\n",
    "from keras.layers import Dropout as DO\n",
    "\n",
    "from keras.layers import LSTM#, sequential\n",
    "from keras.optimizers import Adam\n",
    "from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, History\n",
    "from keras.preprocessing.image import ImageDataGenerator as ImgDataGen\n",
    "from keras.preprocessing.image import img_to_array, load_img\n",
    "from keras.regularizers import l2\n",
    "from keras.models import Model\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "from keras.utils import to_categorical\n",
    "from keras.utils.vis_utils import plot_model\n",
    "from keras.wrappers.scikit_learn import KerasRegressor\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "from sklearn.model_selection import ShuffleSplit, train_test_split\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "import glob\n",
    "import cv2\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "import pydot\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.pyplot import figure\n",
    "import h5py\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "#-   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -\n",
    "import os\n",
    "from os import listdir\n",
    "from os.path import isfile, join\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "#from keras import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "WIBExGwgBx0o",
    "tags": []
   },
   "outputs": [],
   "source": [
    "path \t\t\t= r'/content/gdrive/My Drive/Colab_Datasets'\n",
    "os.chdir(path)\n",
    "\n",
    "seed \t\t\t= 7\n",
    "np.random.seed(seed)\n",
    "\n",
    "save_dir = os.path.join(os.getcwd(), 'NueralNetworksSavedModels')\n",
    "if not os.path.isdir(save_dir):\n",
    "    os.makedirs(save_dir)\n",
    "    \n",
    "RevNo,i \t\t\t= 0,0 \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5c3-lngRDYBY"
   },
   "source": [
    "# Programme Code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1553
    },
    "id": "cGaEauZWDJ8l",
    "outputId": "6b1160f6-924b-420a-a17a-7497506df97f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TrustworthyCnn_Rev 1\n",
      "Found 561 images.\n",
      "Found 140 images.\n",
      "Found 37 images.\n",
      "Epoch 1/10\n",
      "15/15 [==============================] - 3s 174ms/step - loss: 10747.8418 - mean_absolute_error: 28.9775 - acc: 0.0688 - val_loss: 4.6458 - val_mean_absolute_error: 1.7449 - val_acc: 0.0938\n",
      "Epoch 2/10\n",
      "15/15 [==============================] - 7s 483ms/step - loss: 4.4285 - mean_absolute_error: 1.6905 - acc: 0.0919 - val_loss: 3.5738 - val_mean_absolute_error: 1.5083 - val_acc: 0.1481\n",
      "Epoch 3/10\n",
      "15/15 [==============================] - 2s 129ms/step - loss: 4.1003 - mean_absolute_error: 1.6416 - acc: 0.0936 - val_loss: 4.0743 - val_mean_absolute_error: 1.6583 - val_acc: 0.1111\n",
      "Epoch 4/10\n",
      "15/15 [==============================] - 2s 128ms/step - loss: 4.1987 - mean_absolute_error: 1.6506 - acc: 0.0977 - val_loss: 3.6246 - val_mean_absolute_error: 1.5889 - val_acc: 0.0741\n",
      "Epoch 5/10\n",
      "15/15 [==============================] - 7s 494ms/step - loss: 4.1150 - mean_absolute_error: 1.6266 - acc: 0.1036 - val_loss: 4.0859 - val_mean_absolute_error: 1.6801 - val_acc: 0.1111\n",
      "Epoch 6/10\n",
      "15/15 [==============================] - 2s 127ms/step - loss: 3.7112 - mean_absolute_error: 1.5534 - acc: 0.1099 - val_loss: 4.0301 - val_mean_absolute_error: 1.6041 - val_acc: 0.1094\n",
      "Epoch 7/10\n",
      "15/15 [==============================] - 7s 495ms/step - loss: 3.7073 - mean_absolute_error: 1.5537 - acc: 0.1083 - val_loss: 3.6450 - val_mean_absolute_error: 1.5626 - val_acc: 0.0833\n",
      "Epoch 8/10\n",
      "15/15 [==============================] - 2s 129ms/step - loss: 3.8382 - mean_absolute_error: 1.5906 - acc: 0.0710 - val_loss: 3.3847 - val_mean_absolute_error: 1.5261 - val_acc: 0.1389\n",
      "Epoch 9/10\n",
      "15/15 [==============================] - 2s 126ms/step - loss: 3.1723 - mean_absolute_error: 1.4403 - acc: 0.1067 - val_loss: 4.5233 - val_mean_absolute_error: 1.7527 - val_acc: 0.0463\n",
      "Epoch 10/10\n",
      "15/15 [==============================] - 8s 502ms/step - loss: 3.7105 - mean_absolute_error: 1.5621 - acc: 0.0873 - val_loss: 3.7184 - val_mean_absolute_error: 1.6429 - val_acc: 0.1019\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n8VV1ZVXdW1W37KD9zVV1U/9xS1WNVtVB/bbbq+rf+21Lh1XjE2FHIX7z5s07Pe7jH/94DjzwwGGV\nBQAAQAcNsyf+/UnO2VFja+0drbUTW2snJvm1JP/UWlszsMtZ/faRIdY4dJdcckm+/e1v58QTT8xz\nnvOcnHHGGTnvvPPyrGc9K0nyYz/2YznllFNy7LHH5oorrth63FFHHZVVq1bl9ttvz5IlS/L6178+\nxx57bF784hdn48aN0/V2AAAAmEZDm52+tXZDVR01xd1fm+SqYdWyxf/3d1/L1++6f5ee81mH7Z9L\nX3bsDtvf/va355ZbbslNN92U66+/Pi996Utzyy23bJ3d/corr8xBBx2UjRs35jnPeU5+/Md/PIsW\nLdruHMuWLctVV12VP/3TP82rXvWqfOQjH8kFF1ywS98HAAAAu79pvya+quam12P/kYHNLcmnqurG\nqrroUY6/qKqWVtXSlStXDrPUXeLUU0/d7vZs73znO3PCCSfktNNOy5133plly5ZNOOboo4/OiSee\nmCQ55ZRTcvvttz9R5QIAALAb2R3uE/+yJP88bij9D7fWVlTVIUmurapvtNZumOzg1toVSa5IkpGR\nkbazF9pZj/kTZd68eVuXr7/++nz605/OF7/4xcydOzdnnnnmpLdv23fffbcuz5w503B6AACAvdS0\n98QneU3GDaVvra3oP9+b5Jokp05DXbvEggULsn79+knb1q1bl4ULF2bu3Ln5xje+kS996UtPcHUA\nAAB0ybT2xFfVAUlekOSCgW3zksxora3vL784yVunqcTv26JFi/K85z0vxx13XPbbb78ceuihW9vO\nOeecvOc978mSJUvyjGc8I6eddto0VgoAAMDurlrb6Qj0x3/iqquSnJlkcZJ7klyaZHaStNbe09/n\nwiTntNZeM3DcU9PrfU96PzL8dWvtt6fymiMjI23p0u3vSHfrrbdmyZIl389bYRyfKQAAwK5VVTdO\n5e5sw5yd/rVT2Of96d2KbnDbd5KcMJyqAAAAoLt2h2viAQAAgCkQ4gEAAKAjdodbzO063/xmcuaZ\n22/73/+79zw6mtx228RjFi1KFi9OHnkk+c53JrYffHBy0EHJww8n3/3uxPZDD00OPDDZtCm5446J\n7U9+crL//smGDcmdd05sP/zwZP785IEHkhUrJrYfeWQyd25y//3J9743sf0pT0nmzEnWrk3uuWdi\n+9FHJ/vsk6xZk6xcObH9qU9NZs9OVq1KVq+e2P60pyUzZyb33pvcd19v2913Jxdf3Fu+/vre8+/9\nXvL3f7/9sfvtl3ziE73l3/qt5DOf2b590aLkIx/pLf/aryVf/OL27UcckfzlX/aW3/jG5Kabtm9/\n+tOTK67oLV90UfKtb23ffuKJyeWX95YvuCBZvnz79tNPT/7X/+ot//iPT3z/Z5+d/M//2Vt+yUuS\n8bf2+9EfTd70pt7y+O9dkrzqVckv/ELvf/tzz53YfuGFvceqVclP/MTE9osvTl796t735qd+amL7\nr/xK8rKX9b73P//zE9t/4zeSF76w97m98Y0T23/nd5If+qHkC19Ifv3XJ7ZffnnvM/z0p5O3vW1i\n+3vfmzzjGcnf/V3y+78/sf0v/qL3/f3Qh5J3v3ti+9VX9/7be//7e4/xPv7x3nf/Xe9KPvzhie2+\ne71l372J7b57vWXfvYntvnu9Zd+9ie2+e757ie+e79727dP93dsJPfEAAADQEUObnX46mJ3+ieEz\nBQAA2LWmOju9nvjdzPz585Mkd911V35isiE3Sc4888yM/7FivMsvvzwbNmzYun7uuedm7dq1u65Q\nAAAAnnBC/G7qsMMOy9VXX/24jx8f4j/+8Y/nwAMP3BWlAQAAME2E+CG75JJL8id/8idb19/ylrfk\nbW97W84+++ycfPLJefazn52//du/nXDc7bffnuOOOy5JsnHjxrzmNa/JkiVLcv7552fjwKQHF198\ncUZGRnLsscfm0ksvTZK8853vzF133ZWzzjorZ511VpLkqKOOyqpVq5Ikl112WY477rgcd9xxubw/\nGcPtt9+eJUuW5PWvf32OPfbYvPjFL97udQAAAJh+e9bs9I/mE5ckd//7rj3nk56dvOTtO2x+9atf\nnTe+8Y35xV/8xSTJhz/84Xzyk5/MG97whuy///5ZtWpVTjvttJx33nmpqknP8e53vztz587Nrbfe\nmptvvjknn3zy1rbf/u3fzkEHHZTR0dGcffbZufnmm/OGN7whl112Wa677rosXrx4u3PdeOONed/7\n3pcvf/nLaa3luc99bl7wghdk4cKFWbZsWa666qr86Z/+aV71qlflIx/5SC644IJd8CEBAACwK+iJ\nH7KTTjop9957b+66667827/9WxYuXJgnPelJ+fVf//Ucf/zxeeELX5gVK1bknsluD9d3ww03bA3T\nxx9/fI4//vitbR/+8Idz8skn56STTsrXvva1fP3rX99pPZ///Odz/vnnZ968eZk/f35e8YpX5HOf\n+1yS5Oijj86JJ56YJDnllFNy++23f5/vHgAAgF1p7+qJ30mP+TC98pWvzNVXX5277747r371q/NX\nf/VXWblyZW688cbMnj07Rx11VDZt2vSYz/vd7343v/d7v5evfOUrWbhwYS688MLHdZ4t9t13363L\nM2fONJweAABgN6Mn/gnw6le/Oh/84Adz9dVX55WvfGXWrVuXQw45JLNnz851112XO+64Y6fHP//5\nz89f//VfJ0luueWW3HzzzUmS+++/P/PmzcsBBxyQe+65J5/4xCe2HrNgwYKsX79+wrnOOOOMfPSj\nH82GDRvy4IMP5pprrskZZ5yxC98tAAAAw7J39cRPk2OPPTbr16/P4Ycfnic/+cn5yZ/8ybzsZS/L\ns5/97IyMjOSZz3zmTo+/+OKL87M/+7NZsmRJlixZklNOOSVJcsIJJ+Skk07KM5/5zBx55JF53vOe\nt/WYiy66KOecc04OO+ywXHfddVu3n3zyybnwwgtz6qmnJkle97rX5aSTTjJ0HgAAoAOqtTbdNewy\nIyMjbfz902+99dYsWbJkmiraM/lMAQAAdq2qurG1NvJo+xlODwAAAB0hxAMAAEBHCPEAAADQEUI8\nAAAAdIQQDwAAAB0hxAMAAEBHCPFDtnbt2rzrXe963Mdffvnl2bBhwy6sCAAAgK4S4oesKyG+tZax\nsbGhvw4AAACPnxA/ZJdcckm+/e1v58QTT8yb3/zmJMk73vGOPOc5z8nxxx+fSy+9NEny4IMP5qUv\nfWlOOOGEHHfccfnQhz6Ud77znbnrrrty1lln5ayzzpr03M961rNy/PHH501velOS5J577sn555+f\nE044ISeccEK+8IUvJEkuu+yyHHfccTnuuONy+eWXJ0luv/32POMZz8hP//RP57jjjsudd96ZT33q\nUzn99NNz8skn55WvfGUeeOCBJ+JjAgAAYApmTXcBT7gzz5y47VWvSn7hF5ING5Jzz53YfuGFvceq\nVclP/MT2bddfv9OXe/vb355bbrklN910U5LkU5/6VJYtW5Z/+Zd/SWst5513Xm644YasXLkyhx12\nWP7hH/4hSbJu3boccMABueyyy3Lddddl8eLF25139erVueaaa/KNb3wjVZW1a9cmSd7whjfkBS94\nQa655pqMjo7mgQceyI033pj3ve99+fKXv5zWWp773OfmBS94QRYuXJhly5blAx/4QE477bSsWrUq\nb3vb2/LpT3868+bNy+/+7u/msssuy2/+5m9O4YMFAABg2PTEP8E+9alP5VOf+lROOumknHzyyfnG\nN76RZcuW5dnPfnauvfba/Oqv/mo+97nP5YADDtjpeQ444IDMmTMnP/dzP5e/+Zu/ydy5c5Mkn/3s\nZ3PxxRcnSWbOnJkDDjggn//853P++edn3rx5mT9/fl7xilfkc5/7XJLkKU95Sk477bQkyZe+9KV8\n/etfz/Oe97yceOKJ+cAHPpA77rhjiJ8GAAAAj8Xe1xO/s57zuXN33r548aP2vD+a1lp+7dd+LT//\n8z8/oe2rX/1qPv7xj+c3fuM3cvbZZ++0B3zWrFn5l3/5l3zmM5/J1VdfnT/+4z/OZz/72cdcz7x5\n87ar7UUvelGuuuqqx3weAAAAhk9P/JAtWLAg69ev37r+Iz/yI7nyyiu3Xmu+YsWK3Hvvvbnrrrsy\nd+7cXHDBBXnzm9+cr371q5Mev8UDDzyQdevW5dxzz80f/MEf5N/+7d+SJGeffXbe/e53J0lGR0ez\nbt26nHHGGfnoRz+aDRs25MEHH8w111yTM844Y8I5TzvttPzzP/9zbrvttiS96/S/9a1v7doPBAAA\ngMdt7+uJf4ItWrQoz3ve83LcccflJS95Sd7xjnfk1ltvzemnn54kmT9/fv7yL/8yt912W9785jdn\nxowZmT179tYgftFFF+Wcc87JYYcdluuuu27redevX5+Xv/zl2bRpU1prueyyy5Ikf/iHf5iLLroo\nf/7nf56ZM2fm3e9+d04//fRceOGFOfXUU5Mkr3vd63LSSSfl9ttv367Wgw8+OO9///vz2te+Ng89\n9FCS5G1ve1ue/vSnD/tjAgAAYAqqtTbdNewyIyMjbenSpdttu/XWW7NkyZJpqmjP5DMFAADYtarq\nxtbayKPtZzg9AAAAdIQQDwAAAB0hxAMAAEBH7BUhfk+67n+6+SwBAACmzx4f4ufMmZPVq1cLn7tA\nay2rV6/OnDlzprsUAACAvdIef4u5I444IsuXL8/KlSunu5Q9wpw5c3LEEUdMdxkAAAB7pT0+xM+e\nPTtHH330dJcBAAAA37c9fjg9AAAA7CmEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgI\nIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4A\nAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOGFuKr6sqqureqbtlB\n+5lVta6qbuo/fnOg7Zyq+mZV3VZVlwyrRgAAAOiSYfbEvz/JOY+yz+daayf2H29NkqqameRPkrwk\nybOSvLaqnjXEOgEAAKAThhbiW2s3JFnzOA49NcltrbXvtNYeTvLBJC/fpcUBAABAB033NfGnV9W/\nVdUnqurY/rbDk9w5sM/y/rZJVdVFVbW0qpauXLlymLUCAADAtJrOEP/VJE9prZ2Q5I+SfPTxnKS1\ndkVrbaS1NnLwwQfv0gIBAABgdzJtIb61dn9r7YH+8seTzK6qxUlWJDlyYNcj+tsAAABgrzZtIb6q\nnlRV1V8+tV/L6iRfSXJMVR1dVfskeU2Sj01XnQAAALC7mDWsE1fVVUnOTLK4qpYnuTTJ7CRprb0n\nyU8kubiqNifZmOQ1rbWWZHNV/VKSTyaZmeTK1trXhlUnAAAAdEX1cvOeYWRkpC1dunS6ywAAAIDH\npKpubK2NPNp+0z07PQAAADBFQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAA\nHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHE\nAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAA\nQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI\n8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAA\nANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANAR\nQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwAAAB0hBAPAAAAHSHEAwAAQEcI8QAAANARQjwA\nAAB0xNBCfFVdWVX3VtUtO2j/yaq6uar+vaq+UFUnDLTd3t9+U1UtHVaNAAAA0CXD7Il/f5JzdtL+\n3SQvaK09O8lvJbliXPtZrbUTW2sjQ6oPAAAAOmXWsE7cWruhqo7aSfsXBla/lOSIYdUCAAAAe4Ld\n5Zr4n0vyiYH1luRTVXVjVV20swOr6qKqWlpVS1euXDnUIgEAAGA6Da0nfqqq6qz0QvwPD2z+4dba\niqo6JMm1VfWN1toNkx3fWrsi/aH4IyMjbegFAwAAwDSZ1p74qjo+yZ8leXlrbfWW7a21Ff3ne5Nc\nk+TU6akQAAAAdh/TFuKr6geS/E2Sn2qtfWtg+7yqWrBlOcmLk0w6wz0AAADsTYY2nL6qrkpyZpLF\nVbU8yaVJZidJa+09SX4zyaIk76qqJNncn4n+0CTX9LfNSvLXrbV/HFadAAAA0BXDnJ3+tY/S/rok\nr5tk+3eSnDDxCAAAANi77S6z0wMAAACPQogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACA\njhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDi\nAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAA\noCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOE\neAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAA\nAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgI\nIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4A\nAAA6YqghvqqurKp7q+qWHbRXVb2zqm6rqpur6uSBtp+pqmX9x88Ms04AAADogmH3xL8/yTk7aX9J\nkmP6j4uSvDtJquqgJJcmeW6SU5NcWlULh1opAAAA7OaGGuJbazckWbOTXV6e5P+0ni8lObCqnpzk\nR5Jc21pb01q7L8m12fmPAQAAALDHm+5r4g9PcufA+vL+th1tBwAAgL3WdIf471tVXVRVS6tq6cqV\nK6e7HAAAABia6Q7xK5IcObB+RH/bjrZP0Fq7orU20lobOfjgg4dWKAAAAEy36Q7xH0vy0/1Z6k9L\nsq619r0kn0zy4qpa2J/Q7sX9bQAAALDXmjXMk1fVVUnOTLK4qpanN+P87CRprb0nyceTnJvktiQb\nkvxsv21NVf1Wkq/0T/XW1trOJsgDAACAPd5QQ3xr7bWP0t6S/OIO2q5McuUw6gIAAIAumu7h9AAA\nAMAUCfEAAADQEUI8AAAAdIQQDwAAAB0hxAMAAEBHCPEAAADQEVMK8VX1N1X10qoS+gEAAGCaTDWU\nvyvJf06yrKreXlXPGGJNAAAAwCSmFOJba59urf1kkpOT3J7k01X1har62aqaPcwCAQAAgJ4pD4+v\nqkVJLkzyuiT/muQP0wv11w6lMgAAAGA7s6ayU1Vdk+QZSf4iyctaa9/rN32oqpYOqzgAAABgmymF\n+CTvbK1dN1lDa21kF9YDAAAA7MBUh9M/q6oO3LJSVQur6heGVBMAAAAwiamG+Ne31tZuWWmt3Zfk\n9cMpCQAAAJjMVEP8zKqqLStVNTPJPsMpCQAAAJjMVK+J/8f0JrF7b3/95/vbAAAAgCfIVEP8r6YX\n3C/ur1+b5M+GUhEAAAAwqSmF+NbaWJJ39x8AAADANJjqfeKPSfK/kjwryZwt21trTx1SXQAAAMA4\nU53Y7n3p9cJvTnJWkv+T5C+HVRQAAAAw0VRD/H6ttc8kqdbaHa21tyR56fDKAgAAAMab6sR2D1XV\njCTLquqXkqxIMn94ZQEAAADjTbUn/peTzE3yhiSnJLkgyc8MqygAAABgokftia+qmUle3Vp7U5IH\nkvzs0KsCAAAAJnjUnvjW2miSH34CagEAAAB2YqrXxP9rVX0syf9N8uCWja21vxlKVQAAAMAEUw3x\nc5KsTvKfBra1JEI8AAAAPEGmFOJba66DBwAAgGk2pRBfVe9Lr+d9O621/7LLKwIAAAAmNdXh9H8/\nsDwnyflJ7tr15QAAAAA7MtXh9B8ZXK+qq5J8figVAQAAAJN61FvM7cAxSQ7ZlYUAAAAAOzfVa+LX\nZ/tr4u9O8qtDqQgAAACY1FSH0y8YdiEAAADAzk1pOH1VnV9VBwysH1hVPza8sgAAAIDxpnpN/KWt\ntXVbVlpra5NcOpySAAAAgMlMNcRPtt9Ub08HAAAA7AJTDfFLq+qyqvrB/uOyJDcOszAAAABge1MN\n8f8tycNJPpTkg0k2JfnFYRUFAAAATDTV2ekfTHLJkGsBAAAAdmKqs9NfW1UHDqwvrKpPDq8sAAAA\nYLypDqdf3J+RPknSWrsvySHDKQkAAACYzFRD/FhV/cCWlao6KkkbRkEAAADA5KZ6m7j/keTzVfVP\nSSrJGUkuGlpVAAAAwARTndjuH6tqJL3g/q9JPppk4zALAwAAALY3pRBfVa9L8stJjkhyU5LTknwx\nyX8aXmkAAADAoKleE//LSZ5ZJ2CmAAAgAElEQVST5I7W2llJTkqydueHAAAAALvSVEP8ptbapiSp\nqn1ba99I8ozhlQUAAACMN9WJ7Zb37xP/0STXVtV9Se4YXlkAAADAeFOd2O78/uJbquq6JAck+ceh\nVQUAAABMMNWe+K1aa/80jEIAAACAnZvqNfEAAADANBPiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACA\njhDiAQAAoCOEeAAAAOiIoYb4qjqnqr5ZVbdV1SWTtP9BVd3Uf3yrqtYOtI0OtH1smHUCAABAF8wa\n1omramaSP0nyoiTLk3ylqj7WWvv6ln1aa/99YP//luSkgVNsbK2dOKz6AAAAoGuG2RN/apLbWmvf\naa09nOSDSV6+k/1fm+SqIdYDAAAAnTbMEH94kjsH1pf3t01QVU9JcnSSzw5snlNVS6vqS1X1Yzt6\nkaq6qL/f0pUrV+6KugEAAGC3tLtMbPeaJFe31kYHtj2ltTaS5D8nubyqfnCyA1trV7TWRlprIwcf\nfPATUSsAAABMi2GG+BVJjhxYP6K/bTKvybih9K21Ff3n7yS5PttfLw8AAAB7nWGG+K8kOaaqjq6q\nfdIL6hNmma+qZyZZmOSLA9sWVtW+/eXFSZ6X5OvjjwUAAIC9ydBmp2+tba6qX0ryySQzk1zZWvta\nVb01ydLW2pZA/5okH2yttYHDlyR5b1WNpfdDw9sHZ7UHAACAvVFtn527bWRkpC1dunS6ywAAAIDH\npKpu7M8Lt1O7y8R2AAAAwKMQ4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADo\nCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEe\nAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAA\nOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKI\nBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAA\ngI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q\n4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Q4gEAAKAjhHgAAADoCCEeAAAAOkKIBwAAgI4Yaoiv\nqnOq6ptVdVtVXTJJ+4VVtbKqbuo/XjfQ9jNVtaz/+Jlh1gkAAABdMGtYJ66qmUn+JMmLkixP8pWq\n+lhr7evjdv1Qa+2Xxh17UJJLk4wkaUlu7B9737DqBQAAgN3dMHviT01yW2vtO621h5N8MMnLp3js\njyS5trW2ph/cr01yzpDqBAAAgE4YZog/PMmdA+vL+9vG+/Gqurmqrq6qIx/jsQAAALDXmO6J7f4u\nyVGttePT623/wGM9QVVdVFVLq2rpypUrd3mBAAAAsLsYZohfkeTIgfUj+tu2aq2tbq091F/9sySn\nTPXYgXNc0Vobaa2NHHzwwbukcAAAANgdDTPEfyXJMVV1dFXtk+Q1ST42uENVPXlg9bwkt/aXP5nk\nxVW1sKoWJnlxfxsAAADstYY2O31rbXNV/VJ64Xtmkitba1+rqrcmWdpa+1iSN1TVeUk2J1mT5ML+\nsWuq6rfS+yEgSd7aWlszrFoBAACgC6q1Nt017DIjIyNt6dKl010GAAAAPCZVdWNrbeTR9pvuie0A\nAACAKRLiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACA\njhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDi\nAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAA\noCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOE\neAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAA\nAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgI\nIR4AAAA6QogHAACAjhDiAQAAoCOEeAAAAOgIIR4AAAA6QogHAACAjhDiAQAAoCOGGuKr6pyq+mZV\n3VZVl0zS/v9U1der6uaq+kxVPWWgbbSqbuo/PjbMOgEAAKALZg3rxFU1M8mfJHlRkuVJvlJVH2ut\nfX1gt39NMtJa21BVFyf530le3W/b2Fo7cVj1AQAAQNcMsyf+1CS3tda+01p7OMkHk7x8cIfW2nWt\ntQ391S8lOWKI9QAAAECnDTPEH57kzoH15f1tO/JzST4xsD6nqpZW1Zeq6sd2dFBVXdTfb+nKlSu/\nv4oBAABgNza04fSPRVVdkGQkyQsGNj+ltbaiqp6a5LNV9e+ttW+PP7a1dkWSK5JkZGSkPSEFAwAA\nwDQYZk/8iiRHDqwf0d+2nap6YZL/keS81tpDW7a31lb0n7+T5PokJw2xVgAAANjtDTPEfyXJMVV1\ndFXtk+Q1SbabZb6qTkry3vQC/L0D2xdW1b795cVJnpdkcEI8AAAA2OsMbTh9a21zVf1Skk8mmZnk\nytba16rqrUmWttY+luQdSeYn+b9VlST/0Vo7L8mSJO+tqrH0fmh4+7hZ7QEAAGCvU63tOZeRj4yM\ntKVLl053GQAAAPCYVNWNrbWRR9tvmMPpAQAAgF1IiAcAAICOEOIBAACgI4R4AAAA6AghHgAAADpC\niAcAAICOEOIBAACgI4R4AAAA6AghHgAAADpCiAcAAICOEOIBAACgI4R4AAAA6AghHgAAADpCiAcA\nAICOEOIBAACgI4R4AAAA6AghHgAAADpCiAcAAICOEOIBAACgI4R4AAAA6IhZ010APGE2P5Ss+U6y\nbnmy38JkwZOT+YckM2dPd2UAAABTIsSzZ2ktuf+uZPWyZNWyZPW3ty2vuzNpY+MOqGTewcn+T+6F\n+gVPShYc1n9+8rbtcxclVdPylgAAALYQ4ummTfcnq2/b9li1rBfWV387eWTDtv1mz0sW/WByxEhy\nwmuTRU9LDjwy2bSuF/bX352s/17vcf+KZMWNyYMrJ77ejNnbQv7OAv++C564zwAAANjrCPHsvkY3\nJ2vvGAjotyWrbustP3DPtv1qRnLgDySLjkmOOqMX2hcdkyw+pheuH2sP+uaHe+dff3eyfiDo398P\n+/d+I/n2dclD9088dp/520L91tB/2MRts/b9/j4bAIDxWuv9DbNqWbLqW/2/nb6VPHBvcvgpydHP\n7/2tNP/g6a4U+D4I8Uyv1pIHVw0Mfx/oWb/vu8nY5m377ndQL5g/7UXJ4qf1etUXHZMcdPSuDcWz\n9un11h945M73e+iB3j+UW3v0xwX+O7/cWx99aOKx+x00EO53MIR/3sHJjJm77n0BAHuGRzb15vlZ\n3Q/rq27bFtoHOxlmz+11bux3UPLvVyc3vq+3/ZBje4H+6OcnT/mhZL8Dp+d9AI9Ltdamu4ZdZmRk\npC1dunS6y2Ayj2zcdn36YI/66tt6Q9u3mLlvctBT+yH9mF5QX9x/nnvQ9NX/eLWWbLxvYMj+97Yf\nwr++v/7APROv16+ZyfxDx4X7wR79/vp+C12vDwB7mtZ6l/it+la/Z33ZttC+9j+2/7th/8P7fzM9\nvfd30+Jjen9H7X94MqN/M6rRzcn3bkq++0/Jd29I/uNLyeZNvRGNTz4xOfqMXqj/gdOTfeZNz3uG\nvVxV3dhaG3nU/YR4dpmxseT+5RN71Fff1ptUbtDWf2yOGQjrT0sOOHLv7H0e3dz7h3p8uL//e9tv\n23jfxGNnzZkY7re7br+/vM/cJ/59AQA7t/nhcb3qy7Y9Hhro6Jg1p3+5YD+sb7l0cNHTkn3nP47X\nfShZ/pXku5/rhfrlX0nGHunNA3TEyLae+iOe4zJAeIII8QzPxrXbB/TVy3o962u+3ftFd4t9Fmzr\nUV98zLZr1Rf9oF94H69HNm0L+DsL/IOT+22x7wE77s3fMrR//qFuuQcAu1pryYbVAyF94Hr1++5I\n2ui2fRc8eVsnx+Knbwvt+x+xrVd9GB5+sNc7/90beo/v3dTr7Z81J/mB0/rX0z8/OeykZKYrcmEY\nhHi+P5sfTu67fWD4+0Dv+uDs7TUzWXjUtl+CB3vX5x9imPd0aK13Pdz4Cfm2u26/3zY450CSrbfc\nG399/vjr9ucuGu4fEgDQRaOPJGu+2w/pAz3qq76VbFq7bb+Z+24bhTi+V33O/tNX/6CNa5M7vtAL\n9Ld/Lrnnlt72fRb0rqPf0lN/6HH+JoBdRIjn0Q3OYDr+Vm3jfxWed/C2IVyD16ovPErPbVeNjfV6\nBQZ79CcL/Du85d6T8qgz8e+7wA85AOx5NqzZvld9y99Pa767/d9P8w/th/Rx16t38fLBB1f1wvyW\nnvrVt/W277ewN+P9llC/+On+7YfHSYhnm4cfHAjptw3MBP/t5OH12/abtV9/yPvAZHJbhr+btXTv\ntd0t98YP4R/o2R+8bm+L2fPGhfuBHv2tQ/iflMye88S/Lxi2sdHej2D3D9y54pENSar3B27N2La8\ndVuN2zZjXPv4Y2Zs+2N5p+07Oj6P45gZveMetabJ2h9vneOXp3b8aEs2t5bNoy2bx1rSBnZJUlXZ\nEjV622r7Mvvrg/v2ShFQ9gqjj/Q6Nbb2qg/MAr9xzbb9Zu6THPSD2wL61p71pyVzDpi++odt3Yrt\nQ/2W+Y/mH7ot0B/9/F6HDzAlQvzeZmy0N1Pp+B711d9O7l8xsGP1fv2d7Fr1wRlM4bHacsu9HfXo\nbwkyO7rl3tbe/B0M4Z9/SPd6Ldgzbb3rxCS3lhz8sWuyu04wbUZb5cHsl/va/KzN/KztP9/X5mdd\n/3ntQNt9/ef7Mzctk//bOGnA7/84Mri+9YeBgX0zeOwk58l2PyBM/EFh6+tP9TUmq2e7Hya2f41s\nd8zkP3hM9j539BozZ1QWzt0nh+y/bw5ZsG8OWTBnu+X99pnG/3/fsGbb9emDs8Cv+c72l53NO3hc\nr3r/evUDn+Lfp9Z6l2FuCfTfvSF58N5e24E/0LuW/ujn92bA3/+waS0VdmdC/J5qw5pxE8r1l9d8\nJxl9eNt+cw7Y/vqqLb3rBz01mb3f9NXP3m38Lfe2m5BvIBBNesu9GQO33Dtsx4HfLfeYgtZaxloy\nOtYy1lpGx1pGW8vYQw+m9b+Ttf7u1AO95xkP3p2ZD9ydmQ/ek1kP3p0Zk/wY9cg+B+bhuYfmof0O\nyUNzDsmm/Q7NxjkHZ9OcQ/Lgvodkwz4H5+GZ+2Xz6FhGx8YyOjqWR0bHMjY6ms1jYxkbHcvmsbGM\njo5mbGwsm0fHes9joxndekxLG+vtPzo2lrGxlrHRzb36+9va2FhGx0Z7723r+pb9t6y3jI2NprWk\n0jIjLZXWD2Rblrdtm5GxXltaMn7//nrGrdd2+2w7ftaM9B6VzJyRzOw/b91WyYzqrc+sZGa1zKzK\nrBmtvz7Y3ovXM2ds2a/XtuW4LW0zksysscwe3ZB9H1mXfR9Zm30fuX/b8+b1/fc2yXcllU2z9s+m\nWQdk0+z9s3HWAVvXN846IBtn7d97zNw/G2YdkI0zD8iGWfvn4Rlz06qy5e+s1h8I0Htu2fLnV2tt\nwvYt69myPklbS2+l9c/R23vwdXrnzeCxO3uNCbVkXO2DNW9/nkw47/bnSev997XmgYdz7/qHeqMi\n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\nEv8PTNuUJ/NkrnsAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1224x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\ncOxM/uLo6bpjAAAAMIIU9A00OVHy0svnXEUHAADguinoG2jH1ET6VZWzL5+rOwoAAAAjRkHfQDum\nVr9W/szyawv6rbfemiQ5fPhwvuM7vuOSx7/zne/MxV8nd7Gf/MmfzJkzZ86/fs973pPjx4/faGQA\nAACGhIK+gbZNlGybaOSlSxT0da9//evziU984obf4+KC/pnPfCbNZvOGzwcAAMBwUNBv0Ec+8pH8\n9E//9PnXP/qjP5of+7Efy/d+8Nvy7r/2V3Pffffl05/+9GuOe/rpp/OmN70pSfLSSy/lO7/zO3Pv\nvffmfe97X1566aXz+33f931fHnjggbzxjW/Mj/zIjyRJfuqnfiqHDx/Ou971rrzrXe9Kktx11115\n/vnnkyQf/ehH86Y3vSlvetOb8pM/+ZPn3+/ee+/N3//7fz9vfOMb883f/M2veh8AAACGw2TdATbE\nb34kOfLoxp5zz33Jt/z4ZTd/8IMfzA/90A/l+7//+5Mkv/qrv5rf+q3fynd9z4M5XU3ljm3L+bq3\n/5f5tm/7tpRSLnmOn/mZn8mOHTvy+OOP55FHHsmb3/zm89t+7Md+LK973ety7ty5fOM3fmMeeeSR\n/MAP/EA++tGP5rOf/Wx27979qnN97nOfyy/8wi/kj//4j1NVVb72a782f+2v/bW0Wq188YtfzMc/\n/vF87GMfywc+8IF88pOfzIc+9KEN+EUCAABgo7iCfoN6vV6+8pWv5PDhw/n85z+fVquVPXv25H/9\nn3403/FNb883f9M35dChQ3nuuecue47f//3fP1+U77///tx///3nt/3qr/5q3vzmN6fX6+ULX/hC\n/vRP//SKef7gD/4g73vf+7Jz587ceuut+fZv//b8x//4H5Mkd999d7rdbpLkLW95S55++umb++EB\nAADYcONxBf0KV7oH6f3vf38+8YlP5MiRI/ngBz+YX/qlX8qLx47lVz7ze3n962bytq85kLNnz173\nef/iL/4iP/ETP5E/+ZM/SavVynd/93ff0HnWbd++/fzyxMSEIe4AAABDyBX0m/DBD34wv/Irv5JP\nfOITef/735/FxcXceecduXXHdH73P/yHfPnLX77i8V//9V+fX/7lX06SPPbYY3nkkUeSJCdOnMjO\nnTszOzub5557Lr/5m795/piZmZmcPHnyNed6xzvekX/37/5dzpw5k9OnT+dTn/pU3vGOd2zgTwsA\n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\n+uvzz6/dv7pg31fON5EqE6+sy+ox/bK2b157nvV1r+zzyjGry6+c58JjUkqqVKmq1d8H55+z/rp6\n5fX5bdVl9n1l+8XbcuF51l5Wl1h3qfe+cFuy+ke2rP03L1n97/7K61d+H1+436VeN0qSC/ZvNFaP\nLxf8OVg95tWvz79vqjSqlTSqfhpZyUTVT6lWMlGdSyPnUqpzF2w/l0Z1LqV/bm15ZW376rpSraSx\n9jrVyup+1er6UvWTaiWN/uo5019J6a+cX159XLh84bor7XOV19XFx6+sXlFprP5+TmNi9f/TZWJ1\n3fnliQu2ry9Pri03Ljpufd3kRfte6j0uOvcV973WPDeb/Qr7jtDfLcAGOLdylVK8Xp4v3uelS5Tn\nC0ryxf/OPrd8Y/m27Ugmp1/7b+upHcnO3a+sm7zltf9mnty++hfx+t8F515O+i+v/sz9lbXlly/a\nfsHza/ZdWf3Zlk9dYt8Lzt9fWTtu7fyprvpjbqhrKvMXfiCxtu1yx71m3yt8UHHxvl/1jcktrc39\n+QesXPgPvWH1wAMPVA899FDdMQaj37/g07m1/8Esn37t/4hePv3a/1EtX2Ldy5c69sz15yoT13H1\n/8IPBK7yocCFHxpMTG3oP9SeOnoqf/6VU+lXVfpVcq5frS1XOdfP6nJ/bVtVpaqqtX2Sfr/KubV9\nz+/TX9unemWfV53rouNXX/dXC8a55TT6q49ybrXkNvrLmegvrxaV/nrBXSu+/ZczUS1nor9alBvV\ny5msljOxVqYnqpfXCvR6ib6wWK9kW1aXp6qXsy2rr7dlJVPnn1eyLSuZLBt7G8a5qmR57d2X1z7v\nW63h/TQuet7o9x60flVyLquFfq06pp9ywfIFz9Ul1l1quWrkXMrlt1/hnK89f3nNvuu/zhM5l8ms\nP59bXb/+XM5dtH31v9Hq9nOXOP6i85QLz3fx8f1sK8MxrO/laiLnVj8eWEvaWHueyLnSePXrTORc\nmVj79Xzl9fpyvzTOr1v/gKhfJlf3L5OrHxilkYlSrR1R5fzRpX/JPxOT5//r9TNRrf7alfXt1Svb\n1reXCz7MaKT/qtery/21DzP6KZv9D7XrVJWLiv8Fhb7czIcLpfHKBwAXL+fCdRfvd8G21+y3kefa\n6P3qeM+r7Xfxr9EFz5f7/bD+d2y1+gFmf+3fpet/l69vT/XK373V2vbVdXllXf8yx+a177H6YWhe\n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\nnHAKqzHM/gAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1224x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df \t\t\t        = pd.read_csv('Train_baseball.csv')\n",
    "df['Image_Name'] \t= df['Image_Name'] + '.jpg'\n",
    "test_size \t\t    = 0.05\n",
    "validationSplit \t= 0.2\n",
    "df_trn, df_tst \t\t= train_test_split(df, test_size=test_size, random_state=seed)\n",
    "df_subtrn, df_val \t= train_test_split(df_trn, test_size=validationSplit, random_state=seed)\n",
    "'-------------------------------------------------------------------------------'\n",
    "'Generator Parameters'\n",
    "interpolationMethod \t= 'nearest'                                             #Interpolation method used to resample the image if the target size is different from that of the loaded image.Supported methods are `\"nearest\"`, `\"bilinear\"`, and `\"bicubic\"`\n",
    "'----'\n",
    "dataType \t\t        = 'float32'\n",
    "shuffle \t\t        = True\n",
    "class_mode \t\t        = 'other'\n",
    "inputDirectory     \t    = './TrainPictures'\n",
    "color_mode \t \t        = \"rgb\"\n",
    "has_ext \t\t        = False\n",
    "'Visuals'\n",
    "model_type \t\t        = 'TrustworthyCnn_Rev %d' % (RevNo)\n",
    "model_name \t\t        = 'CNN_%s_model.{epoch:03d}.h5' % model_type\n",
    "print(model_type)\n",
    "monitor \t\t        = 'val_acc'\n",
    "verbose \t\t        = 1\n",
    "save_best_only \t\t    = True\n",
    "\n",
    "filepath \t\t        = os.path.join(save_dir, model_name)\n",
    "\n",
    "################################################################################\n",
    "RevNo \t\t\t    = i + 1\n",
    "outputDirectory    \t= './AugmentedPics/Rev1'\n",
    "'----'\n",
    "test_size \t\t    = 0.05\n",
    "validationSplit \t= 0.2\n",
    "\n",
    "xColName            = 'Image_Name'\n",
    "yColName     \t   \t= 'Trustworthy_Score'\n",
    "\n",
    "img_height         \t= 128\n",
    "img_width          \t= 85\n",
    "img_channels \t\t= 3\n",
    "reshapeSize        \t= (img_width, img_height)\n",
    "input_shape \t\t= (img_width, img_height, img_channels)                      # (batch_size, imageside1, imageside2, channels)\n",
    "\n",
    "A1,A2,A3,A4     \t= 'relu','relu','relu','relu'\n",
    "K1,K2,K3,K4         = (2,2),        (4,4),(3,3),(4,4)\n",
    "S1,S2,S3,S4         = (1,1),        (1,1),(1,1),(1,1)\n",
    "F1,F2,F3,F4         = 32,            32,32,64\n",
    "P1,P2,P3            = (2,2),        (2,2),(2,2)\n",
    "DO1,DO2             = 0.5,0.5\n",
    "U1,U2               = 128,1\n",
    "DA1,DA2             = 'relu','linear'\n",
    "\n",
    "rescale                             = (1./255.)\n",
    "rotation_range                      = 5\n",
    "featurewise_std_normalization \t\t= False \n",
    "samplewise_std_normalization        = True\n",
    "'------------------------------'\n",
    "#zca_epsilon\n",
    "#zca_whitening\n",
    "#width_shift_range\n",
    "#height_shift_range\n",
    "#brightness_range\n",
    "#shear_range\n",
    "#horizontal_flip\n",
    "#vertical_flip\n",
    "\n",
    "'Compiling Parameters'\n",
    "epochs \t\t\t    = 10\n",
    "batch_size \t\t    = 32\n",
    "steps_per_epoch \t= 15                 \n",
    "valStepsPerEpoch \t= 4\n",
    "tstStepsPerEpoch \t= len(df_tst)\n",
    "loss \t\t\t    = 'mean_squared_error'\n",
    "metrics\t            = ['mae', 'acc']                                             # metrics.mae, metrics.acc                                                     \t \t \t\n",
    "optimizer \t\t    = 'adagrad' \n",
    "\n",
    "def cnn1():\n",
    "        cnn = Sequential()\n",
    "        cnn.add(C2D(filters= F1, kernel_size= K1, activation= A1, input_shape=input_shape,strides= S1))\n",
    "        cnn.add(MP2D(pool_size=P1))\n",
    "        cnn.add(DO(DO1))\n",
    "                \n",
    "        #cnn.add(C2D(filters= F2, kernel_size= K2, activation= A2,strides= S2))\n",
    "        #cnn.add(C2D(filters= F3, kernel_size= K3, activation= A3))\n",
    "        #cnn.add(MP2D(pool_size=P2))\n",
    "        #cnn.add(DO(DO2))\n",
    "        \n",
    "        #cnn.add(MP2D(pool_size=P3))\n",
    "\n",
    "        #cnn.add(C2D(filters= F4, kernel_size= K4, activation= A4))\n",
    "                \n",
    "        cnn.add(Flatten())\n",
    "                \n",
    "        cnn.add(Dense(units = U1, activation= DA1))\n",
    "        cnn.add(Dense(units = U2, activation= DA2))\n",
    "                \n",
    "        cnn.compile(loss= loss, optimizer= optimizer,metrics=metrics)\n",
    "        return cnn\n",
    "\n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "    \n",
    "\n",
    "    \n",
    "    \n",
    "    \n",
    "#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),\n",
    "#                               cooldown=0,\n",
    "#                               patience=5,\n",
    "#                               min_lr=0.5e-6)\n",
    "#checkpoint = ModelCheckpoint(filepath=filepath,\n",
    "#                             monitor = monitor,\n",
    "#                             verbose = verbose,\n",
    "#                             save_best_only = save_best_only)\n",
    "#callbacks = [checkpoint, lr_reducer]    \n",
    "    \n",
    "\n",
    "'#%% configuring data-generators'\n",
    "dg_subtrn \t= ImgDataGen(rescale= rescale,\\\n",
    "\t\t\t             validation_split= validationSplit,\\\n",
    "\t\t\t             dtype = dataType,\\\n",
    "\t\t\t             featurewise_std_normalization = featurewise_std_normalization,\\\n",
    "                         rotation_range = rotation_range)\n",
    "dg_tst  \t= ImgDataGen(rescale= rescale,\\\n",
    "\t\t                  dtype = dataType,\\\n",
    "\t\t                  featurewise_std_normalization = featurewise_std_normalization,\\\n",
    "                          rotation_range = rotation_range)\n",
    "\n",
    "\n",
    "'PreProcessing Training Dataset'\n",
    "df_subtrn = dg_subtrn.flow_from_dataframe(dataframe = df_trn,\\\n",
    "\t\t\t\t  directory = inputDirectory,\\\n",
    "\t\t\t\t  x_col = xColName,\\\n",
    "\t\t\t\t  y_col = yColName,\\\n",
    "\t\t\t\t  batch_size = batch_size,\\\n",
    "\t\t\t\t  seed = seed,\\\n",
    "\t\t\t\t  shuffle = True,\\\n",
    "\t\t\t\t  color_mode = color_mode,\\\n",
    "\t\t\t\t  class_mode = class_mode,\\\n",
    "\t\t\t\t  target_size = reshapeSize,\\\n",
    "\t\t\t\t  has_ext = has_ext,\\\n",
    "\t\t\t\t  subset = 'training',\\\n",
    "\t\t\t\t  interpolation = interpolationMethod)#,\\\n",
    "                  #save_to_dir = outputDirectory)\n",
    "'PreProcessing Validation Dataset'\n",
    "df_val = dg_subtrn.flow_from_dataframe(dataframe = df_trn,\\\n",
    "\t\t\t\t\tdirectory = inputDirectory,\\\n",
    "\t\t\t\t\tx_col = xColName,\\\n",
    "\t\t\t\t\ty_col= yColName,\\\n",
    "\t\t\t\t\tbatch_size = batch_size,\\\n",
    "\t\t\t\t\tseed = seed,\\\n",
    "\t\t\t\t\tshuffle = True,\\\n",
    "\t\t\t\t\tcolor_mode = color_mode,\\\n",
    "\t\t\t\t\tclass_mode = class_mode,\\\n",
    "\t\t\t\t\ttarget_size = reshapeSize,\\\n",
    "\t\t\t\t\thas_ext = has_ext,\\\n",
    "\t\t\t\t\tsubset = 'validation',\\\n",
    "\t\t\t\t\tinterpolation = interpolationMethod)\n",
    "'PreProcessing Test Dataset '\n",
    "df_tst1 = dg_tst.flow_from_dataframe(dataframe = df_tst,\\\n",
    "\t\t\t\t\tdirectory=inputDirectory,\\\n",
    "\t\t\t\t\tx_col= xColName,\\\n",
    "\t\t\t\t\ty_col= yColName,\\\n",
    "\t\t\t\t\tbatch_size=batch_size,\\\n",
    "\t\t\t\t\tseed = seed,\\\n",
    "\t\t\t\t\tshuffle=shuffle,\\\n",
    "\t\t\t\t\thas_ext = has_ext,\\\n",
    "\t\t\t\t\tclass_mode=class_mode,\\\n",
    "\t\t\t\t\ttarget_size=reshapeSize,\\\n",
    "\t\t\t\t\tinterpolation = interpolationMethod)\n",
    "\n",
    "model = cnn1()\n",
    "\n",
    "#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),\n",
    "#                               cooldown=0,\n",
    "#                               patience=5,\n",
    "#                               min_lr=0.5e-6)\n",
    "\n",
    "#checkpoint = ModelCheckpoint(filepath=filepath,\n",
    "#                             monitor = monitor,\n",
    "#                             verbose = verbose,\n",
    "#                             save_best_only = save_best_only) \n",
    "\n",
    "history = model.fit_generator(df_subtrn,\\\n",
    "                              steps_per_epoch = steps_per_epoch,\\\n",
    "                              epochs = epochs,\\\n",
    "                              validation_data = df_val,\\\n",
    "                              validation_steps = valStepsPerEpoch)#,\\\n",
    "                              #callbacks = callbacks)\n",
    "\n",
    "acc_tst = model.evaluate_generator(generator= df_tst1,\n",
    "\t\t\t\t                    steps= tstStepsPerEpoch)\n",
    "\t\t\t\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "'#  \"Accuracy\" '\n",
    "fig = plt.gcf()\n",
    "fig.set_size_inches(17, 9)\n",
    "fig.show()\n",
    "plt.plot(history.history['acc'])\n",
    "plt.plot(history.history['val_acc'])\n",
    "plt.axhline(y=acc_tst[1], color='r', linestyle='--')\n",
    "# -- -- -- --\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'validation','test score'], loc='upper left')\n",
    "plt.show()\n",
    "# fig.savefig('test2png.png', dpi=100)\n",
    "\n",
    "'#  \"MAE\" '\n",
    "fig = plt.gcf()\n",
    "fig.set_size_inches(17, 9)\n",
    "fig.show()\n",
    "plt.plot(history.history['mean_absolute_error'])\n",
    "plt.plot(history.history['val_mean_absolute_error'])\n",
    "#plt.axhline(y=mae_tst[1], color='r', linestyle='--')\n",
    "# -- -- -- --\n",
    "plt.title('MAE')\n",
    "plt.ylabel('Error')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'validation','test score'], loc='upper left')\n",
    "plt.show()\n",
    "\n",
    "# fig.savefig('test2png.png', dpi=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "1EzfhdL1LQti"
   },
   "outputs": [],
   "source": [
    "save_dir = os.path.join(os.getcwd(), '/AugmentedPics/rev1')\n",
    "if not os.path.isdir(save_dir):\n",
    "    os.makedirs(save_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "xJMBU_W-J26x"
   },
   "source": [
    "# Visuals\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "2i4SOHdCHdfb"
   },
   "outputs": [],
   "source": [
    "plot_model(model, to_file='./ResnetModel.png',\\\n",
    "\t   show_shapes=True,\\\n",
    "\t   show_layer_names=True,\\\n",
    "\t   rankdir='LR')\n",
    "# model.summary()"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "CNNRev2.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
