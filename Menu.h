#pragma once
#include <iostream>
#include <string>
using namespace std;

class Menu {
public:
    void exibir();
    void setUsuario(string nomee);
    Menu();
    char lerResposta();

protected:
    string nome;
};