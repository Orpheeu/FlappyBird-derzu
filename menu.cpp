#include <iostream>
#include "../includes/Menu.h"


using namespace std;

void Menu:: exibir(){

cout << "\n<====== FLAPPY BIRD ======>" << endl;
cout << "[1] INICIAR" << endl;
cout << "[0] SAIR" << endl;
cout << "-> ";
}

char Menu:: lerResposta(){
 char resp;

 cin >> resp;

 return resp;
}

void Menu:: setUsuario(string nomee){
nome = nomee;

}

Menu:: Menu(){

};