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
#include <algorithm> 
#include "Menu.h"

using namespace std;
using namespace cv;

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
bool lose = false;
int qtdF = 0;
int pontos = 0;
float velocidade = 2;

Mat pipeTopFull;
Mat pipeBottomFull;
Mat flappy;

int espaco = 150; 
int heightTop1 = 300;
int heightTop2 = 130;
int heightTop3 = 340;
int heightTop4 = 210;
int heightTop5 = 120;
int heightTop6 = 320;

int heightBottom1 = heightTop1 + espaco;
int heightBottom2 = heightTop2 + espaco;
int heightBottom3 = heightTop3 + espaco;
int heightBottom4 = heightTop4 + espaco;
int heightBottom5 = heightTop5 + espaco;
int heightBottom6 = heightTop6 + espaco;

float x1 = 600;
float x2 = 600;
float x3 = 600;
float x4 = 600;
float x5 = 600;
float x6 = 600;

float ultimaPosicaoY = 240;
float velocidadeY = 0;  // Velocidade do movimento no eixo Y
const float ALPHA = 0.35;  // Coeficiente do filtro (quanto menor, mais suave)
const float MAX_VARIACAO = 50.0;  // Máxima variação permitida por frame (em pixels)
bool primeiraDeteccao = true;  // Flag para primeira detecção
const float ESPACAMENTO_CANOS = 200.0;  // Distância entre cada pipe

void playlose(){
    system("aplay ../data/hit.wav >/dev/null 2>&1 &"); 
}

void playwins(){
    system("aplay ../data/point.wav >/dev/null 2>&1 &"); 
}

Mat preparaImagem(Mat& original, double escala) {
    Mat resultado, ampliado;
    resize(original, resultado, Size(), 1.0/escala, 1.0/escala);
    flip(resultado, resultado, 1);
    
    // AMPLIAR o centro para facilitar detecção
    int cropX = resultado.cols * 0.15;  // Remove 15% das laterais
    int cropY = resultado.rows * 0.1;   // Remove 10% de cima e baixo
    
    if(cropX > 0 && cropY > 0 && 
       cropX*2 < resultado.cols && cropY*2 < resultado.rows) {
        Rect roi(cropX, cropY, resultado.cols - 2*cropX, resultado.rows - 2*cropY);
        Mat cropped = resultado(roi);
        resize(cropped, ampliado, resultado.size());
        return ampliado;
    }
    
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

float filtroPassaBaixa(float novaPosicao, float posicaoAnterior, float &velocidade) {
    // Calcula a variação
    float delta = novaPosicao - posicaoAnterior;
    
    // Limita variações abruptas (rejeita outliers)
    if(abs(delta) > MAX_VARIACAO) {
        delta = (delta > 0) ? MAX_VARIACAO : -MAX_VARIACAO;
    }
    
    // Aplica o filtro passa-baixa com suavização adicional da velocidade
    velocidade = velocidade * 0.7 + delta * 0.3;  // Suaviza a velocidade
    
    // Posição filtrada considerando a velocidade suavizada
    float posicaoFiltrada = posicaoAnterior + velocidade * ALPHA;
    
    return posicaoFiltrada;
}

int meuDetectAndDraw(Mat& img, CascadeClassifier& cascade, double scale, int recordeAtual, string nomeRecordista) {
    Mat imagemProcessada = preparaImagem(img, scale);
    Mat imagemCinza = converteParaCinza(imagemProcessada);
    vector<Rect> rostos = buscaRostos(imagemCinza, cascade);
    
    if(!lose) {
        qtdF++;

        // Todos movem ao mesmo tempo, sempre
        x1 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x1, heightTop1 - pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeTopFull, x1, heightTop1 - 2*pipeTopFull.rows); // Segundo cano
        drawTransparency(imagemProcessada, pipeBottomFull, x1, heightBottom1);
        drawTransparency(imagemProcessada, pipeBottomFull, x1, heightBottom1 + pipeBottomFull.rows); // Segundo cano


        x2 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x2, heightTop2 - pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeTopFull, x2, heightTop2 - 2*pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeBottomFull, x2, heightBottom2);
        drawTransparency(imagemProcessada, pipeBottomFull, x2, heightBottom2 + pipeBottomFull.rows);

        x3 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x3, heightTop3 - pipeTopFull.rows); 
        drawTransparency(imagemProcessada, pipeTopFull, x3, heightTop3 - 2*pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeBottomFull, x3, heightBottom3);
        drawTransparency(imagemProcessada, pipeBottomFull, x3, heightBottom3 + pipeBottomFull.rows);

        x4 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x4, heightTop4 - pipeTopFull.rows); 
        drawTransparency(imagemProcessada, pipeTopFull, x4, heightTop4 - 2*pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeBottomFull, x4, heightBottom4);
        drawTransparency(imagemProcessada, pipeBottomFull, x4, heightBottom4 + pipeBottomFull.rows);

        x5 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x5, heightTop5 - pipeTopFull.rows); 
        drawTransparency(imagemProcessada, pipeTopFull, x5, heightTop5 - 2*pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeBottomFull, x5, heightBottom5);
        drawTransparency(imagemProcessada, pipeBottomFull, x5, heightBottom5 + pipeBottomFull.rows);

        x6 -= 4*velocidade; 
        drawTransparency(imagemProcessada, pipeTopFull, x6, heightTop6 - pipeTopFull.rows); 
        drawTransparency(imagemProcessada, pipeTopFull, x6, heightTop6 - 2*pipeTopFull.rows);
        drawTransparency(imagemProcessada, pipeBottomFull, x6, heightBottom6);
        drawTransparency(imagemProcessada, pipeBottomFull, x6, heightBottom6 + pipeBottomFull.rows);

        if(qtdF % 100 == 0) { velocidade += 0.2; }

        if(x1 < 70) { 
            // Encontra o cano mais à direita
            float maiorX = max({x2, x3, x4, x5, x6});
            x1 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        if(x2 < 70) { 
            float maiorX = max({x1, x3, x4, x5, x6});
            x2 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        if(x3 < 70) { 
            float maiorX = max({x1, x2, x4, x5, x6});
            x3 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        if(x4 < 70) { 
            float maiorX = max({x1, x2, x3, x5, x6});
            x4 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        if(x5 < 70) { 
            float maiorX = max({x1, x2, x3, x4, x6});
            x5 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        if(x6 < 70) { 
            float maiorX = max({x1, x2, x3, x4, x5});
            x6 = maiorX + ESPACAMENTO_CANOS;
            pontos++; 
            playwins(); 
        }

        Rect pipeRectTop1 = Rect(x1, 0, pipeTopFull.cols, heightTop1);
        Rect pipeRectTop2 = Rect(x2, 0, pipeTopFull.cols, heightTop2);
        Rect pipeRectTop3 = Rect(x3, 0, pipeTopFull.cols, heightTop3);
        Rect pipeRectTop4 = Rect(x4, 0, pipeTopFull.cols, heightTop4);
        Rect pipeRectTop5 = Rect(x5, 0, pipeTopFull.cols, heightTop5);
        Rect pipeRectTop6 = Rect(x6, 0, pipeTopFull.cols, heightTop6);

        int alturaTela = imagemProcessada.rows;
        Rect pipeRectBottom1 = Rect(x1, heightBottom1, pipeBottomFull.cols, alturaTela - heightBottom1);
        Rect pipeRectBottom2 = Rect(x2, heightBottom2, pipeBottomFull.cols, alturaTela - heightBottom2);
        Rect pipeRectBottom3 = Rect(x3, heightBottom3, pipeBottomFull.cols, alturaTela - heightBottom3);
        Rect pipeRectBottom4 = Rect(x4, heightBottom4, pipeBottomFull.cols, alturaTela - heightBottom4);
        Rect pipeRectBottom5 = Rect(x5, heightBottom5, pipeBottomFull.cols, alturaTela - heightBottom5);
        Rect pipeRectBottom6 = Rect(x6, heightBottom6, pipeBottomFull.cols, alturaTela - heightBottom6);

        // DEPOIS de criar os Rects de colisão e ANTES do loop de rostos, adicione:

        if(rostos.size() > 0) {
            Rect r = rostos[0];  // Usa apenas o primeiro rosto
            
            const int FLAPPY_POSICAO_X = 100;
            
            // Posição Y bruta do rosto detectado
            float posicaoYDetectada = r.y + (r.height / 2.0);
            
            // Na primeira detecção, inicializa direto sem filtro
            if(primeiraDeteccao) {
                ultimaPosicaoY = posicaoYDetectada;
                velocidadeY = 0;
                primeiraDeteccao = false;
            } else {
                // Aplica o filtro passa-baixa
                ultimaPosicaoY = filtroPassaBaixa(posicaoYDetectada, ultimaPosicaoY, velocidadeY);
            }
            
            int flappyCentroY = cvRound(ultimaPosicaoY);
            Rect fac = Rect(FLAPPY_POSICAO_X, flappyCentroY - 22, 45, 45);

            // Verificação de colisão
            if(((fac & pipeRectTop1).area() > 3) || ((fac & pipeRectTop2).area() > 3) || 
            ((fac & pipeRectTop3).area() > 3) || ((fac & pipeRectTop4).area() > 3) || 
            ((fac & pipeRectTop5).area() > 3) || ((fac & pipeRectTop6).area() > 3) || 
            ((fac & pipeRectBottom1).area() > 3) || ((fac & pipeRectBottom2).area() > 3) || 
            ((fac & pipeRectBottom3).area() > 3) || ((fac & pipeRectBottom4).area() > 3) || 
            ((fac & pipeRectBottom5).area() > 3) || ((fac & pipeRectBottom6).area() > 3))
            {
                if (!lose) playlose();
                lose = true;
            }

            int drawY = flappyCentroY - 25;
            
            if(FLAPPY_POSICAO_X >= 0 && FLAPPY_POSICAO_X < imagemProcessada.cols - 50 && 
            drawY >= 0 && drawY < imagemProcessada.rows - 50) {
                drawTransparency(imagemProcessada, flappy, FLAPPY_POSICAO_X, drawY);
            }
        } else {
            // Se não detectar rosto, mantém posição mas reduz velocidade gradualmente
            velocidadeY *= 0.95;  // Amortecimento
            ultimaPosicaoY += velocidadeY * ALPHA;
            
            const int FLAPPY_POSICAO_X = 100;
            int flappyCentroY = cvRound(ultimaPosicaoY);
            int drawY = flappyCentroY - 25;
            
            if(FLAPPY_POSICAO_X >= 0 && FLAPPY_POSICAO_X < imagemProcessada.cols - 50 && 
            drawY >= 0 && drawY < imagemProcessada.rows - 50) {
                drawTransparency(imagemProcessada, flappy, FLAPPY_POSICAO_X, drawY);
            }
        }

        string pontuacao = "Pontuacao: " + to_string(pontos);
        putText(imagemProcessada, pontuacao, Point(10, 80), 
                FONT_HERSHEY_PLAIN, 1.8, Scalar(255,255, 0), 2);

        string txtRecorde = "Recorde: " + nomeRecordista + " - " + to_string(recordeAtual);
        putText(imagemProcessada, txtRecorde, Point(10, 40), 
                FONT_HERSHEY_PLAIN, 1.8, Scalar(255,215,0), 2);

    } else {
        // Cria uma imagem PRETA do zero
        Mat telaGameOver = Mat::zeros(imagemProcessada.size(), imagemProcessada.type());
        
        // GAME OVER (vermelho, grande, centralizado)
        putText(telaGameOver, "GAME OVER", Point(90, 200), 
                FONT_HERSHEY_DUPLEX, 3.5, Scalar(0, 0, 255), 4);
        
        // Sua pontuação (branco)
        string txtPontos = "Sua Pontuacao: " + to_string(pontos);
        putText(telaGameOver, txtPontos, Point(150, 270), 
                FONT_HERSHEY_PLAIN, 2.5, Scalar(255,255,255), 2);
        
        // Verifica se bateu recorde
        if(pontos >= recordeAtual) {
            putText(telaGameOver, "NOVO RECORDE!", Point(140, 320), 
                    FONT_HERSHEY_DUPLEX, 2, Scalar(0,255,0), 3);
        } else {
            string txtRecorde = "Recorde: " + nomeRecordista + " - " + to_string(recordeAtual);
            putText(telaGameOver, txtRecorde, Point(110, 320), 
                    FONT_HERSHEY_PLAIN, 2, Scalar(255,215,0), 2);
        }
        
        // Instruções
        putText(telaGameOver, "Pressione qualquer tecla para continuar", 
                Point(60, 390), FONT_HERSHEY_PLAIN, 1.8, Scalar(200,200,200));
        
        imshow("result", telaGameOver);  // Mostra a tela preta
        return 1;
    }

    imshow("result", imagemProcessada);
    return 1;
}

int main(int argc, const char** argv) {
    cout << "=== INICIANDO FLAPPY BIRD ===" << endl;
    
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

    // Carrega imagens AQUI no main, uma única vez
    cout << "Carregando recursos..." << endl;
    pipeTopFull = cv::imread("../data/pipe_up.png", IMREAD_UNCHANGED);
    pipeBottomFull = cv::imread("../data/pipe_bottom.png", IMREAD_UNCHANGED);
    flappy = cv::imread("../data/flappy.png", IMREAD_UNCHANGED);

    if(!stream.is_open() || texto.empty()) {
        cout << "Arquivo de recordes nao encontrado, sera criado." << endl;
        recorde = 0;
        usuarioTopo = "Ninguem";  // Valor padrão
    }
    
    if(pipeTopFull.empty()) {
        cerr << "ERRO: Nao conseguiu carregar ../data/pipe_up.png" << endl;
        return -1;
    }
    if(pipeBottomFull.empty()) {
        cerr << "ERRO: Nao conseguiu carregar ../data/pipe_bottom.png" << endl;
        return -1;
    }
    if(flappy.empty()) {
        cerr << "ERRO: Nao conseguiu carregar ../data/flappy.png" << endl;
        return -1;
    }
    cout << "Imagens carregadas com sucesso!" << endl;

    stream.open("records.txt", ios_base::in);
    if(!stream.is_open()) {
        cout << "Arquivo de recordes nao encontrado, sera criado." << endl;
        recorde = 0;
    } else {
        getline(stream, texto);
        size_t pos = texto.find_first_of(' ');
        if (pos != string::npos) {
            usuarioTopo = texto.substr(0, pos);  
            numeracao = texto.substr(pos + 1);
        }
        stream.close();
        try {
            recorde = stoi(numeracao);
        } catch (...) {
            recorde = 0;
        }
    }

    cascadeName = "haarcascade_frontalface_default.xml";
    scale = 1;

    cout << "Carregando detector facial..." << endl;
    if (!cascade.load(cascadeName)) {
        cerr << "ERRO: Nao foi possivel carregar: " << cascadeName << endl;
        cerr << "Procure em: /usr/share/opencv4/haarcascades/" << endl;
        return -1;
    }
    cout << "Detector carregado!" << endl;

    cout << "Abrindo webcam..." << endl;
    if(!capture.open(0)) {
        cerr << "ERRO: Nao foi possivel abrir a camera!" << endl;
        return 1;
    }
    cout << "Webcam aberta!" << endl;

    while(1) {
        menu.exibir();
        resp = menu.lerResposta();
        
        if(resp == '1')
        {
            lose = false;
            pontos = 0;
            qtdF = 0;
            velocidade = 1;
            x1 = 600;
            x2 = 600 + 200;  // 200 pixels depois
            x3 = 600 + 400;
            x4 = 600 + 600;
            x5 = 600 + 800;
            x6 = 600 + 1000;
            ultimaPosicaoY = 240;
            velocidadeY = 0;
            primeiraDeteccao = true;  // ADICIONE
            
            cout << "Digite seu nome de usuario:" << endl;
            cin >> nome;
            menu.setUsuario(nome);

            if(capture.isOpened()) {
                while(1) {
                    capture >> frame;
                    if(frame.empty()) break;

                    meuDetectAndDraw(frame, cascade, scale, recorde, usuarioTopo);
                    
                    if(lose) {
                        // FORÇA desenhar a tela de Game Over chamando a função de novo
                        meuDetectAndDraw(frame, cascade, scale, recorde, usuarioTopo);
                        
                        // Agora SIM espera tecla
                        while(true) {
                            char tecla = (char)waitKey(30);
                            if(tecla != -1) {
                                break;
                            }
                        }
                        break;
                    }
                    
                    char c = (char)waitKey(10);
                    if(c == 27 || c == '0') break;
                }
                destroyAllWindows(); 
                
                if(pontos > recorde) {
                    cout << "Novo Recorde! Salvando..." << endl; 
                    stream.open("records.txt", ios_base::out);
                    novotexto = nome + " " + to_string(pontos);
                    stream << novotexto;
                    stream.close();
                    recorde = pontos;
                }
                
                cout << "\n--- Fim de Jogo ---" << endl;
                cout << "Sua pontuacao final: " << pontos << " pontos." << endl;
            }
        }
        else if(resp == '0') {
            break;
        }
    }

    cout << endl << "ENCERRANDO..." << endl;
    return 0;
}