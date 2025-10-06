#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <fstream>
#include "Menu.h"

using namespace std;
using namespace cv;

// Suas funções implementadas aqui(keury)
void drawTransparency(Mat& quadro, Mat& sobreposicao, int x, int y) {
    for (int i = 0; i < sobreposicao.rows; i++) {
        for (int j = 0; j < sobreposicao.cols; j++) {
            int fx = x + j;
            int fy = y + i;

            if (fx >= quadro.cols || fy >= quadro.rows || fx < 0 || fy < 0)
                continue;

            Vec4b &pxSobre = sobreposicao.at<Vec4b>(i, j);
            Vec3b &pxQuadro = quadro.at<Vec3b>(fy, fx);

            double alpha = pxSobre[3] / 255.0;

            for (int c = 0; c < 3; c++) {
                pxQuadro[c] = saturate_cast<uchar>(pxQuadro[c] * (1.0 - alpha) + pxSobre[c] * alpha);
            }
        }
    }
}

void drawTransRect(Mat& quadro, Scalar cor, double alpha, Rect regiao) {
    for (int i = 0; i < regiao.height; i++) {
        for (int j = 0; j < regiao.width; j++) {
            int fx = regiao.x + j;
            int fy = regiao.y + i;

            if (fx >= quadro.cols || fy >= quadro.rows || fx < 0 || fy < 0)
                continue;

            Vec3b &pxQuadro = quadro.at<Vec3b>(fy, fx);

            for (int c = 0; c < 3; c++) {
                pxQuadro[c] = saturate_cast<uchar>(pxQuadro[c] * (1.0 - alpha) + cor[c] * alpha);
            }
        }
    }
}


// variaveis globais
bool lose = false; // perdeu ou não
int qtdF = 0; // quantidade de frames
int pontos = 0; // pontos
float velocidade = 1; // velocidade do jogo

Mat pipeTopFull = cv::imread("../data/pipe_up.png", IMREAD_UNCHANGED);
Mat pipeBottomFull = cv::imread("../data/pipe_bottom.png", IMREAD_UNCHANGED);
Mat flappy = cv::imread("../data/flappy.png", IMREAD_UNCHANGED);

//tamanho dos pipes original = width:38px e height:242px
// espaço entre os pipes tem que ser de 100 px
int espaco = 100; 
int heightTop1 = 240;
int heightTop2 = 120;
int heightTop3 = 200;
int heightTop4 = 150;
int heightTop5 = 80;
int heightTop6 = 60;

int heightBottom1 = heightTop1 + espaco;
int heightBottom2 = heightTop2 + espaco;
int heightBottom3 = heightTop3 + espaco;
int heightBottom4 = heightTop4 + espaco;
int heightBottom5 = heightTop5 + espaco;
int heightBottom6 = heightTop6 + espaco;

// os canos aparecem em x = 600
float x1 = 600;
float x2 = 600;
float x3 = 600;
float x4 = 600;
float x5 = 600;
float x6 = 600;

void playlose(){
    system("mplayer ../data/hit.wav >/dev/null 2>&1 &"); 
}

void playwins(){
    system("mplayer ../data/point.wav >/dev/null 2>&1 &"); 
}

Mat preparaImagem(Mat& original, double escala) {
    Mat resultado;
    resize(original, resultado, Size(), 1.0/escala, 1.0/escala);
    flip(resultado, resultado, 1);
    return resultado;
}

Mat converteParaCinza(Mat& colorida) {
    Mat cinza;
    cvtColor(colorida, cinza, COLOR_BGR2GRAY);
    equalizeHist(cinza, cinza);
    return cinza;
}

vector<Rect> buscaRostos(Mat& imagem, CascadeClassifier& detector) {
    vector<Rect> rostos;
    detector.detectMultiScale(imagem, rostos, 1.3, 2, 0|CASCADE_SCALE_IMAGE, Size(40, 40));
    return rostos;
}

int meuDetectAndDraw(Mat& img, CascadeClassifier& cascade, double scale) {
    Mat imagemProcessada = preparaImagem(img, scale);
    Mat imagemCinza = converteParaCinza(imagemProcessada);
    vector<Rect> rostos = buscaRostos(imagemCinza, cascade);
    
    if(!lose) {
        qtdF++;

        // Movimenta e desenha canos de cima e de baixo
        if(qtdF > 0) { x1 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x1, heightTop1 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x1, heightBottom1); }
        if(qtdF > 75) { x2 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x2, heightTop2 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x2, heightBottom2); }  
        if(qtdF > 150) { x3 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x3, heightTop3 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x3, heightBottom3); }
        if(qtdF > 225) { x4 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x4, heightTop4 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x4, heightBottom4); }
        if(qtdF > 300) { x5 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x5, heightTop5 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x5, heightBottom5); }
        if(qtdF > 375) { x6 -= 2*velocidade; drawTransparency(imagemProcessada, pipeTopFull, x6, heightTop6 - pipeTopFull.rows); drawTransparency(imagemProcessada, pipeBottomFull, x6, heightBottom6); }

        if(qtdF % 100 == 0) { velocidade += 0.05; }

        // Reseta canos
        if(x1 < -pipeBottomFull.cols) { x1 = 600; pontos++; playwins(); } 
        if(x2 < -pipeBottomFull.cols) { x2 = 600; pontos++; playwins(); }
        if(x3 < -pipeBottomFull.cols) { x3 = 600; pontos++; playwins(); }
        if(x4 < -pipeBottomFull.cols) { x4 = 600; pontos++; playwins(); }
        if(x5 < -pipeBottomFull.cols) { x5 = 600; pontos++; playwins(); }
        if(x6 < -pipeBottomFull.cols) { x6 = 600; pontos++; playwins(); }

        // Canos de cima colisão
        Rect pipeRectTop1 = Rect(x1, 0, pipeTopFull.cols, heightTop1);
        Rect pipeRectTop2 = Rect(x2, 0, pipeTopFull.cols, heightTop2);
        Rect pipeRectTop3 = Rect(x3, 0, pipeTopFull.cols, heightTop3);
        Rect pipeRectTop4 = Rect(x4, 0, pipeTopFull.cols, heightTop4);
        Rect pipeRectTop5 = Rect(x5, 0, pipeTopFull.cols, heightTop5);
        Rect pipeRectTop6 = Rect(x6, 0, pipeTopFull.cols, heightTop6);

        // Canos de baixo colisão
        int alturaTela = imagemProcessada.rows;
        Rect pipeRectBottom1 = Rect(x1, heightBottom1, pipeBottomFull.cols, alturaTela - heightBottom1);
        Rect pipeRectBottom2 = Rect(x2, heightBottom2, pipeBottomFull.cols, alturaTela - heightBottom2);
        Rect pipeRectBottom3 = Rect(x3, heightBottom3, pipeBottomFull.cols, alturaTela - heightBottom3);
        Rect pipeRectBottom4 = Rect(x4, heightBottom4, pipeBottomFull.cols, alturaTela - heightBottom4);
        Rect pipeRectBottom5 = Rect(x5, heightBottom5, pipeBottomFull.cols, alturaTela - heightBottom5);
        Rect pipeRectBottom6 = Rect(x6, heightBottom6, pipeBottomFull.cols, alturaTela - heightBottom6);

        for (size_t i = 0; i < rostos.size(); i++) {
            Rect r = rostos[i]; 
            Rect fac = Rect(cvRound(r.x+(r.width/2) - 22), cvRound(r.y+(r.height/2) - 22), 45, 45);

            if(((fac & pipeRectTop1).area() > 3) || ((fac & pipeRectTop2).area() > 3) || ((fac & pipeRectTop3).area() > 3) || ((fac & pipeRectTop4).area() > 3) || ((fac & pipeRectTop5).area() > 3) || ((fac & pipeRectTop6).area() > 3) || ((fac & pipeRectBottom1).area() > 3) || ((fac & pipeRectBottom2).area() > 3) || ((fac & pipeRectBottom3).area() > 3) || ((fac & pipeRectBottom4).area() > 3) || ((fac & pipeRectBottom5).area() > 3) || ((fac & pipeRectBottom6).area() > 3))
            {
                if (!lose) playlose();
                lose = true;
            }

            int drawX = cvRound(r.x+(r.width/2) - 25);
            int drawY = cvRound(r.y+(r.height/2) - 25);
            
            if(drawX >= 0 && drawX < imagemProcessada.cols - 50 && drawY >= 0 && drawY < imagemProcessada.rows - 50) {
                drawTransparency(imagemProcessada, flappy, drawX, drawY);
            }
        }
        
        putText(imagemProcessada, to_string(pontos), Point(320, 50), FONT_HERSHEY_PLAIN, 3, Scalar(255,255,255));

    } else {

        drawTransRect(imagemProcessada, Scalar(0, 0, 255), 0.5, Rect(0, 0, imagemProcessada.cols, imagemProcessada.rows));
        
        putText(imagemProcessada, "GAME OVER", Point(85, 200), FONT_HERSHEY_PLAIN, 5, Scalar(255,255,255));
        putText(imagemProcessada, "Pressione qualquer tecla para o menu", Point(15, 250), FONT_HERSHEY_PLAIN, 2, Scalar(255,255,255));
        
        x1 = 600; x2 = 600; x3 = 600; x4 = 600; x5 = 600; x6 = 600;
        qtdF = 0;
        velocidade = 1;
    }

    imshow("result", imagemProcessada);
    return 1;
}

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame;
    CascadeClassifier cascade;
    double scale;
    string cascadeName;
    Menu menu;
    char resp;
    string nome;
    
    fstream stream;
    string texto;
    string novotexto;
    int recorde;
    string numeracao = " ";
    string usuarioTopo;

    stream.open("records.txt", ios_base::in);

    if(!stream.is_open( )){
        cout<<"Não foi possível abrir arquivo de recordes.";
    } else {
        getline(stream, texto);
        size_t pos = texto.find_first_of(' ');
        if (pos != string::npos) {
            usuarioTopo = texto.substr(0, pos);  
            numeracao = texto.substr(pos + 1);
        }
        stream.close();
    }
    
    try {
        recorde = stoi(numeracao);
    } catch (const std::invalid_argument& ia) {
        recorde = 0;
    }

    cascadeName = "haarcascade_frontalface_default.xml";
    scale = 1;

    if (!cascade.load(cascadeName)) {
        cerr << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return -1;
    }

    if(!capture.open(0)) 
    {
        cout << "Capture from camera #0 didn't work" << endl;
        return 1;
    }

    while(1)
    {
        menu.exibir();
        resp = menu.lerResposta();
        if(resp == '1')
        {
            lose = false;
            pontos = 0; // Pontuação é resetada aqui, no início do jogo
            cout<< "Digite seu nome de usuario:" << endl;
            cin >> nome;
            menu.setUsuario(nome);

            if( capture.isOpened() ) 
            {
                while (1)
                {
                    capture >> frame;
                    if( frame.empty() ) { break; }

                    meuDetectAndDraw( frame, cascade, scale);
                    
                    if(lose) { 
                        waitKey(500); // Pequena pausa para o jogador ver o Game Over
                        break; 
                    } 
                
                    char c = (char)waitKey(10);
                    if( c == 27 || c == '0' ) { break; }
                }

                destroyAllWindows(); 
                
                if(pontos > recorde)
                {
                    cout << "Novo Recorde! Salvando..." << endl; 
                    stream.open("records.txt", ios_base::out);
                    novotexto = nome + " " + to_string(pontos);
                    stream << novotexto;
                    stream.close();
                    recorde = pontos; // Atualiza o recorde na sessão atual
                }
                
                cout << "\n--- Fim de Jogo ---" << endl;
                cout << "Sua pontuacao final: " << pontos << " pontos." << endl;
            }
            else
            {
                cout << "Erro: Nao foi possivel abrir a camera!" << endl;
            }
        }
        else if(resp == '0')
        {
            break;
        }
    }

    cout << endl << "ENCERRANDO..." << endl;
    return 0;
}