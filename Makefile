NAME = neural.exe
SRCDIR = src
SRCS =	main.cpp Data.cpp NN.cpp Layer.cpp Types.cpp export.cpp

OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

OBJDIR = obj
CXXFLAGS = -Wall -Wextra -Werror -g3 -MMD -MP

.SILENT:

all: $(OBJDIR) $(NAME)

$(OBJDIR):
	if not exist $(OBJDIR) mkdir $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	c++ $(CXXFLAGS) -c $< -o $@

$(NAME): $(OBJS)
	c++ $(CXXFLAGS) $(OBJS) -o $(NAME)

a: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/wine_red.csv Regress

f: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/forest_fires.csv Regress

car: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/automobile.csv Regress

cub: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/cubic_1d.csv Regress

q: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/quadratic_1d.csv Regress

s: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/saddle_2d.csv Regress

sin: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/sin_1d.csv Regress

ss: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/regression/sinsurf_2d.csv Regress


i: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/iris_binary.csv Class

i3: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/Iris.csv Class

t: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/titanic.csv Class

l: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/linear_2d.csv Class

x: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/xor_2d.csv Class

c: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/circles_2d.csv Class

plot:
	python helper/plot.py history.csv graph.csv complexity.csv

clean:
	if exist $(OBJDIR) rmdir /s /q $(OBJDIR)

fclean: clean
	if exist $(NAME) del /q $(NAME)

re: fclean all

gpush: fclean
	git add .
	git commit -m "metrics"
	git push