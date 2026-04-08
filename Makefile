NAME = neural.exe
SRCDIR = src
SRCS =	main.cpp Data.cpp NN.cpp Layer.cpp Types.cpp

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

i: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/iris_binary.csv Class

t: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/titanic.csv Class

l: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/linear_2d.csv Class

x: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/xor_2d.csv Class

c: $(OBJDIR) $(NAME)
	-.\$(NAME) hw3_data/classification/circles_2d.csv Class

clean:
	if exist $(OBJDIR) rmdir /s /q $(OBJDIR)

fclean: clean
	if exist $(NAME) del /q $(NAME)

re: fclean all

gpush: fclean
	git add .
	git commit -m "initial"
	git push