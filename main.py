import tkinter as tk
import pandas as pd
import math
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix

import config

# Main window and associated side menu
class main_window(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent

        # Main Menu
        main_menu = tk.Frame(parent, width=200, height=config.h)
        mm_title = tk.Label(main_menu, text="Main Menu", font=('bold'))
        mm_title.grid(row=0)
        main_menu.grid(row=0, sticky='n')
        import_tab = tk.Button(main_menu, text="1. Import Data", command=self.open_import)
        import_tab.grid(row=1)
        explore_tab = tk.Button(main_menu, text="2. Exploratory Analysis", command=self.open_explore)
        explore_tab.grid(row=2)
        model_tab = tk.Button(main_menu, text="3. Modelling", command=self.open_modelling)
        model_tab.grid(row=3)
        eval_tab = tk.Button(main_menu, text="4. Model Evaluation", command=self.open_eval)
        eval_tab.grid(row=4)

        # Main View Window
        self.view = tk.Frame(self, width=config.w - 200, height=config.h)
        self.view.grid(row=0, column=1)

        # Menu Tabs
        # 1. Import Data Tab
        self.import_data = c_import_data(self.view)
        self.import_data.grid_propagate(0)
        self.import_data.grid(row=0, sticky="nsew", padx=10, pady=10)
        # 2. Exploratory Analysis Tab
        self.explore = c_exploratory(self.view)
        self.explore.grid_propagate(0)
        self.explore.grid(row=0, sticky="nsew", padx=10, pady=10)
        # 3. Modelling Tab
        self.modelling = c_modelling(self.view)
        self.modelling.grid_propagate(0)
        self.modelling.grid(row=0, sticky="nsew", padx=10, pady=10)
        # 4. Model Evaluation Tab
        self.import_data.tkraise()

    def open_import(self):
        self.import_data.tkraise()

    def open_explore(self):
        self.explore.tkraise()

    def open_modelling(self):
        self.modelling.tkraise()

    def open_eval(self):
        self.eval = c_eval(self.view)
        self.eval.grid_propagate(0)
        self.eval.grid(row=0, sticky="nsew", padx=10, pady=10)
        self.eval.tkraise()

class c_import_data(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=config.w - 200, height=config.h)

        # File Path
        tk.Label(self, text="File Path").grid(row=0)
        self.file_path = tk.StringVar()
        tk.Entry(self, textvariable=self.file_path).grid(row=0, column=1)
        self.import_data = tk.Button(self, command=self.get_inputs, text="Import Data")
        self.import_data.grid(row=99, column=99)

        # Options
        tk.Label(self, text="Header").grid(row=2)
        self.header_option = tk.IntVar(self)
        tk.Checkbutton(self, variable=self.header_option).grid(row=2, column=1)

        tk.Label(self, text="Model Data").grid(row=3)
        self.set_data = tk.StringVar(self)
        tk.Entry(self, textvariable=self.set_data).grid(row=3, column=1)

        tk.Label(self, text="Target Data").grid(row=4)
        self.set_target = tk.StringVar(self)
        tk.Entry(self, textvariable=self.set_target).grid(row=4, column=1)

    def get_inputs(self):
        config.header = self.header_option.get()
        name = self.file_path.get()
        if config.header == 1:
            head_var = 0
        else:
            head_var = None

        config.df = pd.read_csv(r'{}'.format(name.strip('\/"')), header=head_var)
        print(config.df)
        # print("File name: %s" % (self.file_name.get()))

        config.data_cols = int(self.set_data.get())
        config.target_col = int(self.set_target.get())

        config.df_data = config.df.iloc[:, :config.data_cols]
        config.df_target = config.df.iloc[:, config.target_col]

class c_exploratory(tk.Frame):
    def __init__(self, parent):
        self.parent = parent
        self.c_output = c_output
        tk.Frame.__init__(self, parent, width=config.w - 200, height=config.h)

        # Summary Statistics
        self.var_summ = tk.IntVar()
        tk.Label(self, text="Summary Statistics").grid(row=0, sticky='W')
        tk.Checkbutton(self, variable=self.var_summ).grid(row=0, column=1)

        # Frequency Histograms
        self.var_freq = tk.IntVar()
        tk.Label(self, text="Frequency Histograms Settings").grid(row=1, sticky='W')
        tk.Label(self, text="Columns").grid(row=2, column=0)
        tk.Label(self, text="Output all").grid(row=3, column=0)
        tk.Entry(self).grid(row=2, column=1)
        tk.Checkbutton(self, variable=self.var_freq).grid(row=3, column=1)

        # Scatterplots
        self.var_scatter = tk.IntVar()
        tk.Label(self).grid(row=4)
        tk.Label(self, text="Scatterplot Settings").grid(row=5, sticky='W')
        tk.Label(self, text="Include Features").grid(row=6, column=0)
        tk.Label(self, text="Include all").grid(row=7, column=0)
        tk.Entry(self).grid(row=6, column=1)
        tk.Checkbutton(self, variable=self.var_scatter).grid(row=7, column=1)

        # Output Button
        tk.Button(self, text="Output", command=self.get_status).grid(row=8, column=1)

    def get_status(self):
        config.summ_status = self.var_summ.get()
        config.freq_status = self.var_freq.get()
        config.scatter_status = self.var_scatter.get()
        self.c_output(self.master.master.view).grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

class c_output(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=config.w - 200, height=config.h)
        nrows = math.ceil(len(config.df_data.columns) / 3)

        selections = []
        output_frames = tk.Frame(self)
        output_frames.grid(row=1)

        if config.summ_status == 1:
            selections.append("Summary Statistics")
            self.summ_frame = tk.Frame(output_frames)
            self.summ_frame.grid(row=0, sticky="nsew")
            tk.Label(self.summ_frame, text="{}".format(config.df_data.describe())).grid(row=0)

        if config.freq_status == 1:
            selections.append("Frequency Histograms")
            self.freq_frame = tk.Frame(output_frames)
            self.freq_frame.grid(row=0, sticky="nsew")
            # Initialise list for assignment of variables
            f_ax = []
            hist_fig = plt.Figure(figsize=(8, 6), dpi=100)

            for pos, colname in enumerate(config.df_data.columns):
                # if  config.df[pos].column.dtype != "category":
                f_ax.append(hist_fig.add_subplot(nrows, 3, pos + 1))
                f_ax[pos].hist(config.df_data.iloc[:, pos])
                f_ax[pos].title.set_text(str(colname))

            hist_canvas = FigureCanvasTkAgg(hist_fig, self.freq_frame)
            hist_canvas.get_tk_widget().grid(row=1)

        if config.scatter_status == 1:
            selections.append("Scatterplots")
            self.scatter_frame = tk.Frame(output_frames)
            self.scatter_frame.grid(row=0, sticky="nsew")
            s_ax = []
            scatter_fig = plt.Figure(figsize=(8, 6), dpi=100)
            subplot_index = 1

            # Calculate max combinations of x/y (nC2)
            scatter_nrows = 0
            for n in range(1, len(config.df_data.columns)):
                scatter_nrows += n
            scatter_nrows = math.ceil(scatter_nrows / 3)

            for pos, colname in enumerate(config.df_data.columns):
                for pos2, colname2 in enumerate(config.df_data.columns[pos+1:]):
                    s_ax.append(scatter_fig.add_subplot(scatter_nrows, 3, subplot_index))
                    s_ax[subplot_index-1].scatter(config.df_data.iloc[:, pos], config.df_data.iloc[:, pos+pos2])
                    s_ax[subplot_index-1].title.set_text("{} vs {}".format(str(colname), colname2))
                    subplot_index += 1

            scatter_canvas = FigureCanvasTkAgg(scatter_fig, self.scatter_frame)
            scatter_canvas.get_tk_widget().grid(row=1)

        self.selection_bar = tk.StringVar(self)
        self.selection_bar.set(selections[-1])
        tk.OptionMenu(self, self.selection_bar, *selections).grid(row=0, sticky="w")
        self.selection_bar.trace('w', self.OutputRaise)

    def OutputRaise(self, *args):
        if self.selection_bar.get() == "Summary Statistics":
            self.summ_frame.tkraise()
        elif self.selection_bar.get() == "Frequency Histograms":
            self.freq_frame.tkraise()
        elif self.selection_bar.get() == "Scatterplots":
            self.scatter_frame.tkraise()

class c_modelling(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=config.w - 200, height=config.h)

        # Main Window Frame
        self.model_frame = tk.Frame(self)

        # Method Selection
        self.method_var = tk.StringVar(self)
        options = ['Classification', 'Regression', 'Clustering']
        self.method_var.set(options[0])

        dropdown = tk.OptionMenu(self.model_frame, self.method_var, *options)
        dropdown_label = tk.Label(self.model_frame, text="Method")
        dropdown_label.grid(row=0, column=0)
        dropdown.grid(row=0, column=1)
        self.method_var.trace('w', self.UpdateModelsMenu)

        # Model Selection
        self.model_var = tk.StringVar(self)
        self.mdropdown = tk.OptionMenu(self.model_frame, self.model_var, value="N/A")
        self.mdropdown_label = tk.Label(self.model_frame, text="Model")
        self.mdropdown_label.grid(row=1, column=0)
        self.mdropdown.grid(row=1, column=1)

        self.model_frame.grid(row=0)

    def UpdateModelsMenu(self, *args):
        if self.method_var.get() == "Classification":
            options_class = ['Nearest Neighbours', 'Naive Bayes']
            self.model_var.set(options_class[0])
            self.mdropdown = tk.OptionMenu(self.model_frame, self.model_var, *options_class)
            self.mdropdown.grid(row=1, column=1)
        elif self.method_var.get() == "Regression":
            options_reg = ['Linear Regression', 'Support Vector Machines']
            self.model_var.set(options_reg[0])
            self.mdropdown = tk.OptionMenu(self.model_frame, self.model_var, *options_reg)
            self.mdropdown.grid(row=1, column=1)

        self.model_var.trace('w', self.UpdateSettingsMenu)

    # Model Settings
    def UpdateSettingsMenu(self, *args):
        ## Classification
        #### Nearest Neighbours
        if self.model_var.get() == "Nearest Neighbours":
            kn_frame = tk.Frame(self)

            self.test_size_var = tk.StringVar(self)
            testsize_label = tk.Label(kn_frame, text="Test Size")
            testsize_entry = tk.Entry(kn_frame, textvariable=self.test_size_var)
            testsize_label.grid(row=0, column=0)
            testsize_entry.grid(row=0, column=1)

            self.user_n_var = tk.StringVar(self)
            k_n_label = tk.Label(kn_frame, text="Number of Neighbours")
            k_n_entry = tk.Entry(kn_frame, textvariable=self.user_n_var)
            k_n_label.grid(row=1, column=0)
            k_n_entry.grid(row=1, column=1)

            output_button = tk.Button(kn_frame, text="Save Settings", command=self.set_model_settings)
            output_button.grid(row=2, column=10)

            kn_frame.grid(row=1, column=0, columnspan=10, sticky="nsew")

        #### Naive Bayes

        ## Regression
        #### Linear regression
        if self.model_var.get() == "Linear Regression":
            self.lreg_frame = tk.Frame(self)

            self.test_size_var = tk.StringVar(self)
            tk.Label(self.lreg_frame, text="Test Size").grid(row=0, column=0)
            tk.Entry(self.lreg_frame, textvariable=self.test_size_var).grid(row=0, column=1)

            linear_model = ["Standard", "Ridge", "Lasso"]
            self.linear_model_var = tk.StringVar(self)
            tk.Label(self.lreg_frame, text="Linear Model").grid(row=1)
            self.lreg_dropdown = tk.OptionMenu(self.lreg_frame, self.linear_model_var, *linear_model).grid(row=1, column=1)
            self.linear_model_var.set(linear_model[0])

            output_button = tk.Button(self.lreg_frame, text="Save Settings", command=self.set_model_settings)
            output_button.grid(row=4, column=10)

            self.linear_model_var.trace('w', self.update_lreg)

            self.lreg_frame.grid(row=1, column=0, columnspan=10, sticky="nsew")

    def update_lreg(self, *args):
        self.alpha_var = tk.StringVar(self)
        self.max_iter_var = tk.StringVar(self)
        if self.linear_model_var.get() == "Ridge":
            tk.Label(self.lreg_frame, text="Alpha").grid(row=2)
            tk.Entry(self.lreg_frame, textvariable=self.alpha_var).grid(row=2, column=1)
        if self.linear_model_var.get() == "Lasso":
            tk.Label(self.lreg_frame, text="Alpha").grid(row=2)
            tk.Entry(self.lreg_frame, textvariable=self.alpha_var).grid(row=2, column=1)
            tk.Label(self.lreg_frame, text="Max Iterations").grid(row=3, column=0)
            tk.Entry(self.lreg_frame, textvariable=self.max_iter_var).grid(row=3, column=1)

    def set_model_settings(self):
        if self.model_var.get() == "Nearest Neighbours":
            config.selected = "Nearest Neighbours"
            config.kn_test_size = float(self.test_size_var.get())
            config.user_n = int(self.user_n_var.get())

        if self.model_var.get() == "Linear Regression":
            config.selected = "Linear Regression"
            config.lreg_test_size = float(self.test_size_var.get())
            config.lreg_model = self.linear_model_var.get()
            if config.lreg_model in ["Ridge", "Lasso"]:
                config.user_alpha = float(self.alpha_var.get())
                if config.lreg_model == "Lasso":
                    config.user_iter = int(self.max_iter_var.get())

class c_eval(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, width=config.w - 200, height=config.h)
        tk.Label(self, text="{}".format(config.selected)).grid(row=0)

        if config.selected == "Nearest Neighbours":
            x_train, x_test, y_train, y_test =\
                train_test_split(config.df_data, config.df_target, test_size=config.kn_test_size, random_state=config.set_seed)
            kn = KNeighborsClassifier(n_neighbors=config.user_n)
            kn.fit(x_train, y_train)
            pred = kn.predict(x_test)

            # Standard accuracy score
            test_score = kn.score(x_test, y_test)
            tk.Label(self, text="Accuracy: {:.2f}".format(test_score)).grid(row=1, sticky="w")

            # K-fold cross validation
            cross_val_params = KFold(n_splits=3, shuffle=True, random_state=config.set_seed)
            cross_val = cross_val_score(kn, config.df_data, config.df_target, cv=cross_val_params)
            tk.Label(self, text="Cross-validation Score: {} \n Average Cross-validation score: {:.2f}"
                     .format(cross_val, cross_val.mean())).grid(row=2, sticky="w")

            # Confusion matrix
            targets = list(set(config.df_target))
            cm = confusion_matrix(y_test, pred, targets)
            tk.Label(self, text="Confusion Matrix: \n {}".format(cm)).grid(row=3, sticky="w")

            cm_fig = plt.figure()
            ax = cm_fig.add_subplot(111)
            ax.set_xticklabels([''] + targets)
            ax.set_yticklabels([''] + targets)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            cax = ax.matshow(cm)
            cm_fig.colorbar(cax)
            cm_canvas = FigureCanvasTkAgg(cm_fig, self)
            cm_canvas.get_tk_widget().grid(row=4)

        if config.selected == "Linear Regression":
            x_train, x_test, y_train, y_test = \
                train_test_split(config.df_data, config.df_target, test_size=config.lreg_test_size, random_state=config.set_seed)

            if config.lreg_model == "Standard":
                lr = LinearRegression().fit(x_train, y_train)
                pred = lr.predict(x_test)

            if config.lreg_model == "Ridge":
                lr = Ridge(alpha=config.user_alpha).fit(x_train, y_train)
                pred = lr.predict(x_test)

            if config.lreg_model == "Lasso":
                lr = Lasso(alpha=config.user_alpha, max_iter=config.user_iter).fit(x_train, y_train)
                pred = lr.predict(x_test)

            # Standard accuracy score
            test_score = lr.score(x_test, y_test)
            tk.Label(self, text="Accuracy: {:.2f}".format(test_score)).grid(row=1, sticky="w")

            # K-fold cross validation
            cross_val_params = KFold(n_splits=3, shuffle=True, random_state=config.set_seed)
            cross_val = cross_val_score(lr, config.df_data, config.df_target, cv=cross_val_params)
            tk.Label(self, text="Cross-validation Score: {} \n Average Cross-validation score: {:.2f}"
                     .format(cross_val, cross_val.mean())).grid(row=2, sticky="w")

if __name__=="__main__":
    app = main_window(None)
    app.mainloop()